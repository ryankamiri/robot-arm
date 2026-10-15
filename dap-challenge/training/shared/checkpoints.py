"""Training-state checkpoints, separate from inference-only Hub policy exports."""

from dataclasses import dataclass
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping

from jaxtyping import UInt8
import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from .config import _plain_config


@dataclass
class CheckpointState:
    """Non-model state returned to the caller after restoring a checkpoint."""

    config: dict[str, Any]
    progress: dict[str, int]
    extra_state: dict[str, Any]


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy legacy, CPU Torch and available GPU RNG state."""
    generator, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    cpu: UInt8[Tensor, "N_rng"] = torch.get_rng_state()
    cuda: list[UInt8[Tensor, "N_rng"]] | None = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    mps: UInt8[Tensor, "N_rng"] | None = (
        torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    )
    return {
        "python": random.getstate(),
        "numpy": {
            "generator": generator,
            "keys": keys.tolist(),
            "position": position,
            "has_gauss": has_gauss,
            "cached_gaussian": cached_gaussian,
        },
        "torch_cpu": cpu,
        "torch_cuda": cuda,
        "torch_mps": mps,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore captured generators; fail if a saved GPU backend is unavailable."""
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("cannot restore CUDA RNG state without CUDA")
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("saved CUDA RNG device count differs from this machine")
    if state["torch_mps"] is not None and not torch.backends.mps.is_available():
        raise ValueError("cannot restore MPS RNG state without MPS")

    numpy_state = state["numpy"]
    random.setstate(state["python"])
    np.random.set_state(
        (
            numpy_state["generator"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if state["torch_mps"] is not None:
        torch.mps.set_rng_state(state["torch_mps"].cpu())


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
        "config": _plain_config(config),
        "progress": dict(progress),
        "extra_state": dict(extra_state) if extra_state is not None else {},
        "rng": capture_rng_state(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:
            temporary_path = Path(file.name)
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


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
