"""BC windows and rollout-level splitting; observation processing is shared."""

from bisect import bisect_right
from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence, TypedDict

from jaxtyping import Float
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from training.shared.episodes import Episode, load_episode
from shared.observations import build_history


class BCSample(TypedDict):
    """One supervised example; DataLoader prepends B to the tensor shapes."""

    images: Float[Tensor, "L I C_in H_in W_in"]
    positions: Float[Tensor, "L A"]
    expert_action: Float[Tensor, "A"]
    episode_id: str
    time_index: int


class BCDataset(Dataset[BCSample]):
    """Map each time index to causal observation history and its sent target."""

    def __init__(self, episodes: Sequence[Episode], history_length: int) -> None:
        if not episodes or any(len(episode) == 0 for episode in episodes):
            raise ValueError("BCDataset requires nonempty episodes")
        if isinstance(history_length, bool) or not isinstance(history_length, int) or history_length < 1:
            raise ValueError("history_length must be a positive integer")

        ids = [episode.episode_id for episode in episodes]
        if len(set(ids)) != len(ids):
            raise ValueError("episode IDs must be unique")
        if any(episode.metadata["joint_names"] != episodes[0].metadata["joint_names"] for episode in episodes):
            raise ValueError("all episodes must use the same joint order")
        if any(tuple(view.shape[1:] for view in episode.frames) != tuple(view.shape[1:] for view in episodes[0].frames) for episode in episodes):
            raise ValueError("all episodes must use the same preprocessed image shape")

        self.episodes = tuple(episodes)
        self.history_length = history_length
        self.ends = []
        total = 0
        for episode in episodes:
            total += len(episode)
            self.ends.append(total)

    def __len__(self) -> int:
        """Return total supervised examples, including padded startup examples."""
        return self.ends[-1]

    def __getitem__(self, index: int) -> BCSample:
        """Return one episode-local window; its target is at the current row."""
        if not 0 <= index < len(self):
            raise IndexError(index)

        episode_index = bisect_right(self.ends, index)
        start = self.ends[episode_index - 1] if episode_index else 0
        time_index = index - start
        episode = self.episodes[episode_index]
        history = build_history(
            episode.frames, episode.frame_indices, episode.positions, time_index, self.history_length
        )
        return BCSample(
            images=history.images,
            positions=history.positions,
            expert_action=episode.actions[time_index].clone(),
            episode_id=episode.episode_id,
            time_index=time_index,
        )


@dataclass
class BCDataLoaders:
    """Loaders and split IDs for recording in config/checkpoint extra_state."""

    train: DataLoader[BCSample]
    validation: DataLoader[BCSample]
    train_episode_ids: tuple[str, ...]
    validation_episode_ids: tuple[str, ...]
    generator: torch.Generator


def create_dataloaders(
    episode_paths: Sequence[str | Path], config: Mapping[str, Any], seed: int
) -> BCDataLoaders:
    """Split whole rollouts before windowing, then load local episode data."""
    settings = {}
    for section, key in (
        ("preprocessing", "image_width"),
        ("preprocessing", "image_height"),
        ("model", "history_length"),
        ("training", "batch_size"),
    ):
        value = config[section][key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")

        settings[key] = value
    fraction = config["dataset"]["validation_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    paths = sorted(Path(path).resolve() for path in episode_paths)
    if len(paths) < 2 or len(set(paths)) != len(paths):
        raise ValueError("provide at least two distinct episode directories")

    random.Random(seed).shuffle(paths)
    validation_count = math.ceil(len(paths) * fraction)
    if validation_count >= len(paths):
        raise ValueError("validation_fraction leaves no training episodes")

    validation_paths, train_paths = paths[:validation_count], paths[validation_count:]
    train_episodes = [load_episode(path, settings["image_width"], settings["image_height"]) for path in train_paths]
    validation_episodes = [load_episode(path, settings["image_width"], settings["image_height"]) for path in validation_paths]
    all_episodes = train_episodes + validation_episodes
    if len({episode.episode_id for episode in all_episodes}) != len(all_episodes):
        raise ValueError("episode IDs must be unique across both splits")
    if any(episode.metadata["joint_names"] != all_episodes[0].metadata["joint_names"] for episode in all_episodes):
        raise ValueError("training and validation must use the same joint order")

    train_dataset = BCDataset(train_episodes, settings["history_length"])
    validation_dataset = BCDataset(validation_episodes, settings["history_length"])
    generator = torch.Generator().manual_seed(seed)
    validation_generator = torch.Generator().manual_seed(seed)
    return BCDataLoaders(
        train=DataLoader(
            train_dataset, batch_size=settings["batch_size"], shuffle=True,
            drop_last=False, num_workers=0, generator=generator,
        ),
        validation=DataLoader(
            validation_dataset, batch_size=settings["batch_size"], shuffle=False,
            drop_last=False, num_workers=0, generator=validation_generator,
        ),
        train_episode_ids=tuple(episode.episode_id for episode in train_episodes),
        validation_episode_ids=tuple(episode.episode_id for episode in validation_episodes),
        generator=generator,
    )
