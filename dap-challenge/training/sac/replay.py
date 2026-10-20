"""Persist scored SAC episodes and sample causal histories without duplicating windows."""

from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import re

from jaxtyping import Bool, Float, UInt8
import torch
from torch import Tensor

from models.dimensions import A
from training.shared.utils import atomic_torch_save, file_sha256


@dataclass
class ScoredEpisode:
    """T requested/sent targets and T+1 preprocessed RGB observations, arm view first."""

    episode_id: str
    policy_version: str
    reward_version: str
    reward: float
    images: UInt8[Tensor, 'N I C_in H_in W_in']
    positions: Float[Tensor, 'N A']
    requested_actions: Float[Tensor, 'T A']
    sent_actions: Float[Tensor, 'T A']
    outcome: str = 'scored'

    def validate(self) -> None:
        """Admit only complete scored attempts with finite normalized controller values."""
        if not isinstance(self.episode_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', self.episode_id):
            raise ValueError('episode_id must be a safe local filename')

        if any(not isinstance(value, str) or not value.strip() for value in (self.policy_version, self.reward_version)) or self.outcome != 'scored':
            raise ValueError('only scored attempts with policy/reward provenance enter replay')

        if not isinstance(self.reward, (int, float)) or not torch.isfinite(torch.tensor(self.reward)):
            raise ValueError('terminal reward must be finite')

        if self.requested_actions.ndim != 2:
            raise ValueError('requested actions must have shape [T,A]')

        count = self.requested_actions.shape[0]
        if count < 1 or self.images.ndim != 5 or self.images.shape[:3] != (count + 1, 2, 3):
            raise ValueError('images must contain T+1 observations [N,2,3,H,W]')

        if self.images.dtype != torch.uint8 or min(self.images.shape[-2:]) < 1:
            raise ValueError('images must be preprocessed uint8 RGB pixels')

        for values, shape in ((self.positions, (count + 1, A)),
                              (self.requested_actions, (count, A)), (self.sent_actions, (count, A))):
            if values.shape != shape or values.dtype != torch.float32:
                raise ValueError('positions/actions must be float32 with matching six-channel shapes')

            if not torch.isfinite(values).all() or (values.abs() > 1).any():
                raise ValueError('positions/actions must be finite normalized values in [-1,1]')

        if any(value.device.type != 'cpu' for value in
               (self.images, self.positions, self.requested_actions, self.sent_actions)):
            raise ValueError('persist episode tensors on CPU')

    def save(self, directory: Path) -> Path:
        """Finalize an immutable scored episode; pending and faulted takes stay elsewhere."""
        self.validate()
        path = directory / f'{self.episode_id}.pt'
        if path.exists():
            raise FileExistsError(path)

        return atomic_torch_save({'format_version': 1, **vars(self)}, path, replace=False)

    @classmethod
    def load(cls, path: Path):
        payload = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
        if payload.pop('format_version', None) != 1:
            raise ValueError('unsupported SAC episode format')

        episode = cls(**payload)
        episode.validate()
        return episode


@dataclass
class ReplayBatch:
    """Current/next causal windows and recorded requested MDP actions."""

    images: Float[Tensor, 'B L I C_in H_in W_in']
    positions: Float[Tensor, 'B L A']
    actions: Float[Tensor, 'B A']
    rewards: Float[Tensor, 'B']
    next_images: Float[Tensor, 'B L I C_in H_in W_in']
    next_positions: Float[Tensor, 'B L A']
    terminated: Bool[Tensor, 'B']


class EpisodeReplay:
    """Uniform transition replay with a bounded episode cache and persistent source files."""

    def __init__(self, history_length: int, reward_version: str, cache_episodes: int):
        if history_length < 1 or cache_episodes < 1 or not reward_version:
            raise ValueError('history/cache sizes and reward version are required')

        self.history_length = history_length
        self.reward_version = reward_version
        self.cache_episodes = cache_episodes
        self.records: list[dict] = []
        self.ends: list[int] = []
        self.cache: OrderedDict[str, ScoredEpisode] = OrderedDict()
        self.image_shape = None

    def __len__(self) -> int:
        return self.ends[-1] if self.ends else 0

    def add(self, path: Path, expected_digest: str | None = None) -> int:
        """Verify immutable provenance and return credits only for newly admitted transitions."""
        path = path.resolve()
        digest = file_sha256(path)
        if expected_digest is not None and digest != expected_digest:
            raise ValueError('persisted replay episode changed since checkpoint')

        episode = ScoredEpisode.load(path)
        for record in self.records:
            if record['episode_id'] == episode.episode_id:
                if record['sha256'] != digest:
                    raise ValueError('duplicate episode ID with different contents')

                return 0
        if episode.reward_version != self.reward_version:
            raise ValueError('episode reward version differs from learner')

        if self.image_shape is not None and episode.images.shape[1:] != self.image_shape:
            raise ValueError('replay episodes must have matching preprocessed camera dimensions')

        self.image_shape = episode.images.shape[1:]
        count = episode.requested_actions.shape[0]
        self.records.append({'episode_id': episode.episode_id, 'path': str(path), 'sha256': digest})
        self.ends.append(len(self) + count)
        return count

    def _episode(self, index: int) -> ScoredEpisode:
        record = self.records[index]
        key = record['episode_id']
        if key not in self.cache:
            path = Path(record['path'])
            if file_sha256(path) != record['sha256']:
                raise ValueError('replay source changed during training')

            self.cache[key] = ScoredEpisode.load(path)
        self.cache.move_to_end(key)
        while len(self.cache) > self.cache_episodes:
            self.cache.popitem(last=False)
        return self.cache[key]

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        """Sample with replacement and pad histories only within their own episode."""
        if not len(self) or batch_size < 1:
            raise ValueError('sampling requires replay and a positive batch size')

        fields = [[None] * batch_size for _ in range(7)]
        selected: dict[int, list[tuple[int, int]]] = {}
        for row, index in enumerate(torch.randint(len(self), (batch_size,)).tolist()):
            episode_index = bisect_right(self.ends, index)
            time_index = index - (self.ends[episode_index - 1] if episode_index else 0)
            selected.setdefault(episode_index, []).append((row, time_index))
        for episode_index, rows_and_times in selected.items():
            episode = self._episode(episode_index)
            for row, time_index in rows_and_times:
                rows = torch.arange(time_index - self.history_length + 1, time_index + 1).clamp_min(0)
                next_rows = torch.arange(time_index - self.history_length + 2, time_index + 2).clamp_min(0)
                terminal = time_index == episode.requested_actions.shape[0] - 1
                values = (episode.images[rows], episode.positions[rows], episode.requested_actions[time_index],
                          torch.tensor(episode.reward if terminal else 0., dtype=torch.float32),
                          episode.images[next_rows], episode.positions[next_rows], torch.tensor(terminal))
                for destination, value in zip(fields, values):
                    destination[row] = value
        tensors = [torch.stack(values).to(device) for values in fields]
        # Current and next image histories (indices 0 and 4): uint8 [0,255] -> float32 [0,1].
        tensors[0] = tensors[0].float() / 255
        tensors[4] = tensors[4].float() / 255
        return ReplayBatch(*tensors)
