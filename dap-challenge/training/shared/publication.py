"""Bounded background actor delivery independent of learner optimization."""

from threading import Condition, Thread

from shared.policy_snapshots import ActorSnapshot, send_actor


class AsyncPolicyPublisher:
    """Send one immutable snapshot at a time and retain only the newest pending version."""

    def __init__(self, url: str, timeout_s: float):
        self.url, self.timeout_s = url, timeout_s
        self.condition = Condition()
        self.pending: ActorSnapshot | None = None
        self.in_flight: str | None = None
        self.acknowledgement: dict | None = None
        self.failure: Exception | None = None
        self.last_identity: tuple[str, int, str] | None = None
        self.closing = False
        self.worker: Thread | None = None

    def snapshot(self) -> dict:
        """Expose delivery state without confusing a queued version with an acknowledged one."""
        with self.condition:
            return {'in_flight': self.in_flight, 'pending': self.pending.version if self.pending else None,
                    'acknowledgement': dict(self.acknowledgement) if self.acknowledgement else None,
                    'error': str(self.failure) if self.failure else None}

    def check(self) -> None:
        """Propagate worker failures on the learner thread at its next polling boundary."""
        with self.condition:
            if self.failure is not None:
                raise RuntimeError(f'Background actor publication failed: {self.failure}') from self.failure

    def submit(self, snapshot: ActorSnapshot) -> dict:
        """Queue bytes without waiting for network I/O or touching live model parameters."""
        with self.condition:
            self.check()
            if self.closing:
                raise RuntimeError('actor publisher is closed')

            identity = (snapshot.run_id, snapshot.actor_updates, snapshot.digest)
            if identity != self.last_identity:
                self.pending = snapshot
                self.last_identity = identity
                if self.worker is None:
                    self.worker = Thread(target=self._run, name='actor-publisher', daemon=True)
                    self.worker.start()
                self.condition.notify_all()
        return {'publication_state': 'queued', 'policy_version': snapshot.version, 'sha256': snapshot.digest}

    def _run(self) -> None:
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending is not None or self.closing)
                if self.pending is None:
                    return

                snapshot = self.pending
                self.pending = None
                self.in_flight = snapshot.version
            try:
                acknowledgement = send_actor(snapshot, self.url, self.timeout_s)
            except Exception as error:
                with self.condition:
                    self.failure = error
                    self.pending = None
                    self.in_flight = None
                    self.condition.notify_all()
                return
            with self.condition:
                self.acknowledgement = acknowledgement
                self.in_flight = None
                self.condition.notify_all()

    def close(self, flush: bool = True, raise_errors: bool = True) -> None:
        """Finish the in-flight send and optionally drain the newest queued snapshot on shutdown."""
        with self.condition:
            self.closing = True
            if not flush:
                self.pending = None
            self.condition.notify_all()
            worker = self.worker
        if worker is not None:
            worker.join(timeout=2 * self.timeout_s + 1)
            if worker.is_alive():
                error = TimeoutError('Actor publisher did not finish within its shutdown deadline.')
                with self.condition:
                    self.failure = error
                if raise_errors:
                    raise error

        if raise_errors:
            self.check()
