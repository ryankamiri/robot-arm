"""Training-state checkpoints, separate from inference-only Hub policy exports."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.optim import Optimizer

from shared.config import plain_config
from .randomness import capture_rng_state, restore_rng_state
from .utils import atomic_torch_save


@dataclass
class CheckpointState:
    """Non-model state returned to the caller after restoring a checkpoint."""

    config: dict[str, Any]
    progress: dict[str, int]
    extra_state: dict[str, Any]


def save_checkpoint(
    path: str | Path,
    models: Mapping[str, nn.Module],
    optimizers: Mapping[str, Optimizer],
    config: Mapping[str, Any],
    progress: Mapping[str, int],
    extra_state: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save state at the caller's explicit path, replacing that path."""
    if not models:
        raise ValueError("at least one named model is required")
    if any(
        not isinstance(key, str)
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for key, value in progress.items()
    ):
        raise ValueError("progress must map string names to nonnegative integer counters")

    payload = {
        "format_version": 1,
        "models": {name: model.state_dict() for name, model in models.items()},
        "model_training": {name: model.training for name, model in models.items()},
        "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        "config": plain_config(config),
        "progress": dict(progress),
        "extra_state": dict(extra_state) if extra_state is not None else {},
        "rng": capture_rng_state(),
    }
    return atomic_torch_save(payload, Path(path))


def load_checkpoint(
    path: str | Path,
    models: Mapping[str, nn.Module],
    optimizers: Mapping[str, Optimizer],
    restore_rng: bool = True,
) -> CheckpointState:
    """Strictly restore into supplied objects, then return config/counters/extra."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format_version")
    if set(models) != set(payload["models"]):
        raise ValueError("model names do not match the checkpoint")
    if set(optimizers) != set(payload["optimizers"]):
        raise ValueError("optimizer names do not match the checkpoint")

    for name, model in models.items():
        model.load_state_dict(payload["models"][name], strict=True)
        model.train(payload["model_training"][name])
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(payload["optimizers"][name])
    if restore_rng:
        restore_rng_state(payload["rng"])
    return CheckpointState(
        config=payload["config"],
        progress=payload["progress"],
        extra_state=payload["extra_state"],
    )
