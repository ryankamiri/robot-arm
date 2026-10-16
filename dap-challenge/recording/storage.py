"""Durable pending takes, timestamped video and explicit training-data promotion."""

from datetime import datetime, timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any
from uuid import uuid4

import av
from jaxtyping import UInt8
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from models.dimensions import A
from training.shared.normalization import JOINT_NAMES


def write_metadata(directory: Path, metadata: dict[str, Any]) -> None:
    """Atomically replace episode metadata without exposing a partial JSON file."""
    temporary = directory / 'metadata.tmp'
    with temporary.open('w') as file:
        json.dump(metadata, file, indent=2, allow_nan=False)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(directory / 'metadata.json')


class EpisodeStore:
    """Keep pending episodes nested away from the training loader's discovery path."""

    def __init__(self, root: Path, recover: bool = True):
        self.root = Path(root).resolve()
        self.pending = self.root / '.pending'
        self.pending.mkdir(parents=True, exist_ok=True)
        if recover:
            self.recover_pending()

    def recover_pending(self) -> None:
        for entry in self.list_pending():
            if entry['status'] == 'recording':
                path = self.path(entry['episode_id'])
                metadata = self.metadata(entry['episode_id'])
                metadata.update(status='incomplete', error='Recorder exited before this take was finalized.')
                metadata['row_count'] = self.recover_rows(path)
                write_metadata(path, metadata)
            elif entry['status'] == 'accepted':
                metadata = self.metadata(entry['episode_id'])
                metadata['status'] = 'review'
                write_metadata(self.path(entry['episode_id']), metadata)

    def path(self, episode_id: str) -> Path:
        if not episode_id or Path(episode_id).name != episode_id or episode_id.startswith('.'):
            raise ValueError('invalid episode ID')

        path = self.pending / episode_id
        if not path.is_dir():
            raise FileNotFoundError(episode_id)

        return path

    def metadata(self, episode_id: str) -> dict[str, Any]:
        return json.loads((self.path(episode_id) / 'metadata.json').read_text())

    def list_pending(self) -> list[dict[str, Any]]:
        episodes = []
        for path in sorted(self.pending.glob('*/metadata.json')):
            try:
                metadata = json.loads(path.read_text())
                episodes.append({key: metadata.get(key) for key in (
                    'episode_id', 'status', 'created_at', 'row_count', 'error', 'diagnostics',
                )})
            except (OSError, ValueError):
                continue
        return episodes

    def recover_rows(self, path: Path) -> int:
        """Recover complete journal rows while preserving damaged or partial originals."""
        rows = []
        journal = path / 'rows.jsonl'
        if not journal.exists():
            return 0

        for line in journal.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                break
        if rows:
            pq.write_table(pa.Table.from_pylist(rows), path / 'steps.parquet')
        return len(rows)

    def keep(self, episode_id: str) -> Path:
        path = self.path(episode_id)
        metadata = self.metadata(episode_id)
        if metadata['status'] != 'review' or metadata.get('row_count', 0) < 1:
            raise ValueError('only finalized, complete takes can be kept')
        if not (path / 'video.mp4').is_file() or not (path / 'steps.parquet').is_file():
            raise ValueError('take is missing video or robot data')

        destination = self.root / episode_id
        if destination.exists():
            raise ValueError('episode already exists in training data')

        metadata['status'] = 'accepted'
        write_metadata(path, metadata)
        try:
            path.rename(destination)
        except OSError:
            metadata['status'] = 'review'
            write_metadata(path, metadata)
            raise
        return destination

    def discard(self, episode_id: str) -> None:
        shutil.rmtree(self.path(episode_id))


class EpisodeWriter:
    """Preserve calibrated values and the exact frames associated with sent commands."""

    def __init__(self, store: EpisodeStore, device_metadata: dict[str, Any], fps: int):
        self.episode_id = 'episode_' + uuid4().hex
        self.path = store.pending / self.episode_id
        self.path.mkdir()
        self.started = time.monotonic()
        self.metadata = {
            **device_metadata,
            'schema_version': 2,
            'episode_id': self.episode_id,
            'status': 'recording',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'joint_names': list(JOINT_NAMES),
            'normalized': False,
            'controller_convention': 'lerobot_range',
            'target_iterations_per_second': fps,
            'video': {'codec': 'h264', 'encoder': 'libx264', 'crf': 18, 'variable_timestamps': True},
            'row_count': 0,
        }
        write_metadata(self.path, self.metadata)
        self.fps = fps
        self.rows: list[dict[str, Any]] = []
        self.container = None
        self.stream = None
        self.last_sequence = None
        self.frame_index = -1
        self.frame_time = 0.0
        self.frame_origin = None
        self.journal = (self.path / 'rows.jsonl').open('a')
        self.timing = (self.path / 'timing.jsonl').open('a')
        self.diagnostics = {'iterations': 0, 'missed_deadlines': 0, 'repeated_frames': 0,
                            'max_cycle_s': 0.0, 'max_camera_age_s': 0.0}

    def append(
        self, rgb: UInt8[np.ndarray, 'H_src W_src C_in'], sequence: int,
        camera_time: float, observation_time: float, action_time: float,
        positions: list[float], actions: list[float],
    ) -> None:
        if camera_time < self.started:
            return
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('camera must produce RGB uint8 [H_src,W_src,C_in]')
        if len(positions) != A or len(actions) != A or not np.isfinite(positions + actions).all():
            raise ValueError('measured positions and sent targets must be finite [A] values')
        if not self.started <= camera_time <= observation_time <= action_time:
            raise ValueError('frame, observation and command timestamps must be chronological')
        if self.last_sequence is not None and (sequence < self.last_sequence or camera_time < self.frame_time):
            raise ValueError('camera frame sequence and timestamps must not go backward')

        if sequence != self.last_sequence:
            if self.container is None:
                self.container = av.open(str(self.path / 'video.mp4'), 'w', options={
                    'movflags': 'frag_every_frame+empty_moov+default_base_moof',
                })
                self.stream = self.container.add_stream('libx264', rate=self.fps)
                self.stream.width, self.stream.height = rgb.shape[1], rgb.shape[0]
                self.stream.pix_fmt = 'yuv420p'
                self.stream.time_base = Fraction(1, 1_000_000)
                self.stream.codec_context.time_base = Fraction(1, 1_000_000)
                self.stream.options = {'crf': '18', 'bf': '0', 'g': str(self.fps), 'tune': 'zerolatency'}
                self.frame_origin = camera_time
                self.metadata['camera']['width'] = rgb.shape[1]
                self.metadata['camera']['height'] = rgb.shape[0]
            frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            frame.time_base = Fraction(1, 1_000_000)
            frame.pts = round((camera_time - self.frame_origin) * 1_000_000)
            for packet in self.stream.encode(frame):
                self.container.mux(packet)
            self.frame_index += 1
            self.last_sequence = sequence
            self.frame_time = camera_time
        else:
            self.diagnostics['repeated_frames'] += 1
        row = {
            'time_index': len(self.rows), 'frame_index': self.frame_index,
            'timestamp_s': observation_time - self.started,
            'camera_timestamp_s': self.frame_time - self.started,
            'action_timestamp_s': action_time - self.started,
            'positions': positions, 'actions': actions,
        }
        self.journal.write(json.dumps(row, allow_nan=False) + '\n')
        self.journal.flush()
        os.fsync(self.journal.fileno())
        self.rows.append(row)

    def record_timing(self, duration: float, camera_age: float) -> None:
        self.diagnostics['iterations'] += 1
        self.diagnostics['missed_deadlines'] += int(duration > 1 / self.fps)
        self.diagnostics['max_cycle_s'] = max(self.diagnostics['max_cycle_s'], duration)
        self.diagnostics['max_camera_age_s'] = max(self.diagnostics['max_camera_age_s'], camera_age)
        self.timing.write(json.dumps({'cycle_s': duration, 'camera_age_s': camera_age}) + '\n')
        self.timing.flush()

    def finish(self, error: str | None = None) -> None:
        """Close streams and leave every take pending until explicitly accepted."""
        try:
            if self.container is not None:
                try:
                    for packet in self.stream.encode():
                        self.container.mux(packet)
                finally:
                    self.container.close()
            if self.rows:
                pq.write_table(pa.Table.from_pylist(self.rows), self.path / 'steps.parquet')
        except Exception as failure:
            error = '; '.join(part for part in (error, str(failure)) if part)
        finally:
            self.journal.close()
            self.timing.close()
            self.metadata.update(
                status='incomplete' if error or not self.rows else 'review',
                error=error or ('No observations recorded.' if not self.rows else None),
                row_count=len(self.rows), diagnostics=self.diagnostics,
            )
            write_metadata(self.path, self.metadata)
