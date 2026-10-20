"""Shared training filesystem utilities for atomic writes and exclusive run ownership."""

from contextlib import contextmanager
import fcntl
import hashlib
import os
import shutil
from pathlib import Path
import tempfile
from typing import Any

import torch


def file_sha256(path: Path) -> str:
    """Hash a file without allocating a second copy of its contents."""
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def atomic_torch_save(payload: Any, path: Path, replace: bool = True) -> Path:
    """Replace a local archive only after its complete contents reach disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as file:
            temporary = Path(file.name)
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
        if replace:
            temporary.replace(path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def atomic_copy(source: Path, destination: Path) -> Path:
    """Expose an already complete archive atomically without changing its byte identity."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.tmp', delete=False) as file:
            temporary = Path(file.name)
            with source.open('rb') as original:
                shutil.copyfileobj(original, file)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


@contextmanager
def training_directory(path: Path):
    """Prevent two learners from writing checkpoints and publication versions into one run."""
    path.mkdir(parents=True, exist_ok=True)
    with (path / '.learner.lock').open('a') as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another learner owns this output directory') from error

        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)
