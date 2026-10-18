"""Exclusive hardware lease shared by BC recording and autonomous execution."""

import fcntl
from pathlib import Path
from threading import Lock


class HardwareLease:
    def __init__(self, path: Path):
        self.path = path
        self.lock = Lock()
        self.owner = None
        self.file = None

    def acquire(self, owner: str) -> None:
        with self.lock:
            if self.owner is not None:
                raise ValueError(f'Hardware is owned by {self.owner}; disconnect that session first.')

            self.path.parent.mkdir(parents=True, exist_ok=True)
            file = self.path.open('a')
            try:
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                file.close()
                raise ValueError('Another backend owns the robot hardware.') from error

            self.owner, self.file = owner, file

    def release(self, owner: str) -> None:
        with self.lock:
            if self.owner == owner:
                self.file.close()
                self.owner = self.file = None
