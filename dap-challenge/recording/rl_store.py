"""Durable RL attempts kept separate from BC demonstrations and admitted only after scoring."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from queue import Queue, Full, Empty
import re
import shutil
from threading import Thread
import time
from uuid import uuid4

from jaxtyping import Float, UInt8
import torch
from torch import Tensor

from recording.storage import CameraVideo, write_metadata
from shared.devices import CameraFrame
from shared.normalization import JOINT_NAMES, normalize_controller
from shared.observations import preprocess_image
from shared.rewards import score_answers
from training.sac.replay import ScoredEpisode
from training.shared.utils import atomic_torch_save


def safe_identifier(value: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', value):
        raise ValueError('Use letters, numbers, hyphens or underscores for local IDs.')

    return value


class RolloutStore:
    def __init__(self, root: Path, rubric: dict):
        self.root, self.rubric = root.resolve(), deepcopy(rubric)
        for kind in ('pending', 'scored', 'aborted'):
            (self.root / kind).mkdir(parents=True, exist_ok=True)
        self.recover()

    def path(self, episode_id: str) -> Path:
        safe_identifier(episode_id)
        for kind in ('pending', 'scored', 'aborted'):
            path = self.root / kind / episode_id
            if path.is_dir() and not path.is_symlink():
                return path
        raise FileNotFoundError(episode_id)

    def recover(self) -> None:
        for path in (self.root / 'pending').iterdir():
            if not path.is_dir() or path.is_symlink():
                continue
            try:
                metadata = json.loads((path / 'metadata.json').read_text())
            except (OSError, ValueError):
                continue
            if metadata['status'] in ('recording', 'finalizing'):
                metadata.update(status='aborted', error='Backend stopped before final observation and videos were finalized.')
                write_metadata(path, metadata)
                path.rename(self.root / 'aborted' / path.name)
            elif metadata['status'] == 'scored':
                self.commit_score(path, metadata)
        for path in (self.root / 'scored').glob('*/metadata.json'):
            self.commit_score(path.parent, json.loads(path.read_text()))

    def episodes(self) -> list[dict]:
        return [json.loads(path.read_text()) for kind in ('pending', 'scored', 'aborted')
                for path in sorted((self.root / kind).glob('*/metadata.json'))]

    def begin(self, tasks: list[str], policy_version: str, hardware: dict) -> 'AttemptWriter':
        if not tasks or len(tasks) != len(set(tasks)) or not set(tasks).issubset(self.rubric['tasks']):
            raise ValueError('Select known behaviors before starting an attempt.')

        episode_id = 'attempt_' + uuid4().hex
        directory = self.root / 'pending' / episode_id
        directory.mkdir()
        metadata = {'episode_id': episode_id, 'status': 'recording', 'selected_tasks': tasks,
                    'policy_version': policy_version, 'rubric': self.rubric, 'hardware': deepcopy(hardware),
                    'created_at': datetime.now(timezone.utc).isoformat(), 'transitions': 0}
        write_metadata(directory, metadata)
        return AttemptWriter(self, directory, metadata)

    def commit_score(self, path: Path, metadata: dict) -> None:
        destination = self.root / 'scored' / path.name
        if path != destination:
            path.rename(destination)
        archive = destination / 'episode.pt'
        if not archive.is_file():
            raise ValueError('Scored attempt has no complete replay archive.')

        # A hard link exposes the immutable archive to the learner only after metadata and videos commit.
        inbox = self.root / 'scored' / f'{path.name}.pt'
        if not inbox.exists():
            os.link(archive, inbox)
        if (destination / 'observations').is_dir():
            shutil.rmtree(destination / 'observations')

    def score(self, episode_id: str, cued: list[str], answers: dict, notes: str) -> dict:
        path = self.path(episode_id)
        metadata = json.loads((path / 'metadata.json').read_text())
        submission = {'cued_tasks': cued, 'answers': answers, 'notes': notes}
        if metadata['status'] == 'scored':
            if metadata['submission'] != submission:
                raise ValueError('A scored attempt is immutable; its reward cannot change within this run.')

            return metadata
        if metadata['status'] != 'review':
            raise ValueError('Only finalized complete attempts can be scored.')

        if metadata['rubric'] != self.rubric:
            raise ValueError('Attempt rubric differs from the immutable run definition.')

        result = score_answers(self.rubric, metadata['selected_tasks'], cued, answers)
        rows = [json.loads(line) for line in (path / 'steps.jsonl').read_text().splitlines()]
        images: UInt8[Tensor, 'N I C_in H_in W_in'] = torch.stack([torch.load(chunk, weights_only=True) for chunk in sorted((path / 'observations').glob('*.pt'))])
        positions: Float[Tensor, 'N A'] = torch.tensor([row['positions'] for row in rows], dtype=torch.float32)
        requested: Float[Tensor, 'T A'] = torch.tensor([row['requested'] for row in rows[:-1]], dtype=torch.float32)
        sent: Float[Tensor, 'T A'] = torch.tensor([row['sent'] for row in rows[:-1]], dtype=torch.float32)
        episode = ScoredEpisode(episode_id, metadata['policy_version'], self.rubric['version'], result['reward'], images,
                                normalize_controller(positions, JOINT_NAMES), normalize_controller(requested, JOINT_NAMES),
                                normalize_controller(sent, JOINT_NAMES))
        (path / 'episode.pt').unlink(missing_ok=True)
        (path / f'{episode_id}.pt').unlink(missing_ok=True)
        episode.save(path)
        (path / f'{episode_id}.pt').rename(path / 'episode.pt')
        metadata.update(status='scored', submission=submission, **result)
        write_metadata(path, metadata)
        self.commit_score(path, metadata)
        return metadata


class AttemptWriter:
    """A bounded background queue keeps encoding, preprocessing and disk writes off the motor loop."""

    def __init__(self, store: RolloutStore, directory: Path, metadata: dict):
        self.store, self.directory, self.metadata = store, directory, metadata
        self.queue = Queue(maxsize=8)
        self.error = None
        self.abort_reason = None
        self.finished = False
        self.count = 0
        self.final_requested = False
        self.final_observation = None
        self.thread = Thread(target=self.run, name='rl-attempt-writer', daemon=True)
        self.thread.start()

    def transition(self, arm: CameraFrame, mac: CameraFrame, positions: list[float], observed: float,
                   requested: list[float], sent: list[float], action_timestamp: float) -> None:
        if self.error:
            raise RuntimeError(f'RL recording failed: {self.error}')

        event = (arm, mac, list(positions), observed, list(requested), list(sent), action_timestamp)
        try:
            self.queue.put_nowait(event)
        except Full as error:
            self.abort_reason = 'RL recording could not keep up; no transitions were dropped into scored replay.'
            raise RuntimeError(self.abort_reason) from error

    def finish(self, observation: tuple | None = None, error: str | None = None) -> None:
        if self.finished:
            return

        self.finished = True
        self.abort_reason = error or self.abort_reason
        # Finalization is queued after the hold command; the motor worker never waits for video encoding.
        self.final_observation = observation
        self.final_requested = True

    def run(self) -> None:
        started = time.monotonic()
        arm_video = CameraVideo(self.directory / 'video.mp4', 30, started)
        mac_video = CameraVideo(self.directory / 'mac_video.mp4', 30, started)
        chunks = self.directory / 'observations'
        chunks.mkdir()
        journal = (self.directory / 'steps.jsonl').open('w')
        origin = None
        closed_videos = set()
        try:
            while True:
                try:
                    event = self.queue.get(timeout=.05)
                except Empty:
                    if not self.final_requested:
                        continue
                    event = None
                if event is None:
                    if self.final_observation is not None and not self.abort_reason:
                        self.write_observation(self.final_observation, chunks, journal, arm_video, mac_video, origin, True)
                    break
                if origin is None:
                    origin = event[3]
                self.write_observation(event, chunks, journal, arm_video, mac_video, origin, False)
            for video in (arm_video, mac_video):
                video.close()
                closed_videos.add(video)
            if self.abort_reason or self.count < 2:
                raise ValueError(self.abort_reason or 'Attempt contains no complete transition.')

            self.metadata.update(status='review', transitions=self.count - 1, final_observation=True)
            write_metadata(self.directory, self.metadata)
        except Exception as error:
            self.error = self.abort_reason or str(error)
            self.metadata.update(status='aborted', transitions=self.count, error=self.error)
            write_metadata(self.directory, self.metadata)
            self.directory.rename(self.store.root / 'aborted' / self.directory.name)
        finally:
            journal.close()
            for video in (arm_video, mac_video):
                if video not in closed_videos and video.container is not None:
                    try:
                        video.close()
                    except Exception:
                        pass

    def write_observation(self, event: tuple, chunks: Path, journal, arm_video: CameraVideo,
                          mac_video: CameraVideo, origin: float, final: bool) -> None:
        arm, mac, positions, observed = event[:4]
        rotation = self.metadata['hardware']['camera']['rotation_clockwise']
        pixels: Float[Tensor, 'I C_in H_in W_in'] = torch.stack([preprocess_image(torch.from_numpy(arm.rgb), 224, 168, rotation),
                              preprocess_image(torch.from_numpy(mac.rgb), 224, 168, 0)])
        atomic_torch_save((pixels * 255).round().clamp(0, 255).to(torch.uint8), chunks / f'{self.count:09d}.pt')
        arm_video.append(arm)
        mac_video.append(mac)
        row = {'time_index': self.count, 'timestamp_s': observed - origin, 'positions': positions,
               'arm_timestamp_s': arm.timestamp - origin, 'mac_timestamp_s': mac.timestamp - origin,
               'arm_frame_index': arm_video.frame_index, 'mac_frame_index': mac_video.frame_index,
               'final': final}
        if not final:
            row.update(requested=event[4], sent=event[5], action_timestamp_s=event[6] - origin)
        journal.write(json.dumps(row, allow_nan=False) + '\n')
        journal.flush()
        os.fsync(journal.fileno())
        self.count += 1
