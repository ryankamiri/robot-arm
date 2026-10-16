"""Seed, capture and restore random generators for BC and future RL training."""

import random
from typing import Any, Mapping

from jaxtyping import UInt8
import numpy as np
import torch
from torch import Tensor


def seed_training(seed: int) -> None:
    """Initialize Python, NumPy and Torch generators without promising determinism."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


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
