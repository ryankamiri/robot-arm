"""Serialize device ownership and recording commands on one continuous worker."""

from concurrent.futures import Future, ThreadPoolExecutor
import fcntl
from copy import deepcopy
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
import time
from typing import Any, Callable

from shared.devices import Devices, SO101Devices, inspect_camera
from shared.ownership import HardwareLease
from recording.storage import EpisodeStore, EpisodeWriter


class RecorderSession:
    def __init__(self, root: Path, factory: Callable[[dict[str, Any]], Devices] = SO101Devices):
        self.store = EpisodeStore(root, recover=False)
        self.lease = HardwareLease(root.parent / 'local' / 'hardware.lock')
        self.factory = factory
        self.next_tick_at = 0.0
        self.devices: Devices | None = None
        self.device_metadata = {}
        self.writer: EpisodeWriter | None = None
        self.finalizer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="take-finalizer")
        self.finalization = None
        self.stop_response = None
        self.commands = Queue()
        self.stop_event = Event()
        self.lock = Lock()
        self.state = {'phase': 'disconnected', 'connected': False, 'error': None,
                      'active_episode_id': None, 'device_mode': None, 'recorded_rows': 0,
                      'diagnostics': {'iterations': 0, 'missed_deadlines': 0,
                                      'max_cycle_s': 0.0, 'camera_age_s': 0.0}}
        self.cleanup_failed = False
        self.owner_file = None
        self.thread = Thread(target=self._run, daemon=True, name='robot-owner')

    def start(self) -> None:
        self.owner_file = (self.store.root / '.recorder.lock').open('a')
        try:
            fcntl.flock(self.owner_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.owner_file.close()
            self.owner_file = None
            raise RuntimeError('Another recorder owns this data directory; use one backend worker.') from error
        self.store.recover_pending()
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise TimeoutError('device worker did not stop; do not start a second hardware session')

        self.finalizer.shutdown(wait=True)
        if self.owner_file is not None:
            self.owner_file.close()
            self.owner_file = None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            state = deepcopy(self.state)
        state['pending'] = self.store.list_pending()
        return state

    def _update(self, **values: Any) -> None:
        with self.lock:
            self.state.update(values)

    def request(self, command: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.thread.is_alive() or self.stop_event.is_set():
            raise RuntimeError('recorder worker is not running')

        future = Future()
        self.commands.put((command, payload or {}, future))
        return future.result(timeout=30)

    def _command(self, command: str, payload: dict[str, Any]) -> None:
        self._complete_finalization()
        if command == 'connect':
            if self.devices is not None or self.cleanup_failed:
                raise ValueError('already connected or prior cleanup failed; inspect status')

            self.lease.acquire('BC recorder')
            try:
                devices = self.factory(payload)
                self.devices = devices
                self.device_metadata = devices.connect()
            except Exception as error:
                self._fault(error)
                raise
            self.next_tick_at = time.monotonic()
            self._update(connected=True, phase='ready', error=None,
                         device_mode=self.device_metadata['device_mode'])
        elif command == 'start':
            if self.devices is None or self.state['phase'] != 'ready':
                raise ValueError('finish reviewing/resetting and connect devices before recording')

            self.writer = EpisodeWriter(self.store, deepcopy(self.device_metadata), fps=30)
            self._update(phase='recording', active_episode_id=self.writer.episode_id, recorded_rows=0)
        elif command == 'stop':
            if self.writer is None:
                raise ValueError('no recording is running')

            self._begin_finalization()
        elif command in ('keep', 'discard'):
            if self.writer is not None or self.finalization is not None:
                raise ValueError('wait for recording finalization before reviewing a take')

            if command == 'keep':
                self.store.keep(payload['episode_id'], payload.get('name', ''), payload.get('notes', ''))
            else:
                self.store.discard(payload['episode_id'])
            self._update(phase='reset' if self.devices is not None else 'disconnected', active_episode_id=None)
        elif command == 'ready':
            if self.devices is None or self.state['phase'] != 'reset':
                raise ValueError('reset is only available after review in a connected session')

            self._update(phase='ready', active_episode_id=None)
        elif command == 'disconnect':
            if self.writer is not None:
                self._begin_finalization('Session ended during recording; take is incomplete.')
            self._disconnect()
            self._update(phase='disconnected', connected=False, active_episode_id=None)
        elif command == 'inspect_camera':
            if self.devices is not None:
                raise ValueError('disconnect before inspecting camera settings')

            self.lease.acquire('camera inspection')
            try:
                self._update(camera_profile=inspect_camera(payload['camera_index']))
            finally:
                self.lease.release('camera inspection')
        else:
            raise ValueError('unknown recorder command')

    def _disconnect(self) -> None:
        devices, self.devices = self.devices, None
        self._update(connected=False)
        if devices is not None:
            try:
                devices.disconnect()
            except Exception:
                self.cleanup_failed = True
                raise
        if not self.cleanup_failed:
            self.lease.release('BC recorder')

    def _begin_finalization(self, error: str | None = None) -> None:
        writer, self.writer = self.writer, None
        self.finalization = self.finalizer.submit(writer.finish, error)
        self._update(phase='finalizing')

    def _complete_finalization(self) -> None:
        if self.finalization is None or not self.finalization.done():
            return

        finished, self.finalization = self.finalization, None
        try:
            finished.result()
            if self.state['phase'] != 'error':
                self._update(phase='review' if self.devices is not None else 'disconnected')
            if self.stop_response is not None:
                self.stop_response.set_result(self.snapshot())
        except Exception as error:
            self._fault(error)
            if self.stop_response is not None:
                self.stop_response.set_exception(error)
        finally:
            self.stop_response = None

    def _fault(self, error: Exception) -> None:
        message = str(error)
        if self.writer is not None:
            self._begin_finalization(message)
        try:
            self._disconnect()
        except Exception as failure:
            message += f'; {failure}'
        self._update(phase='error', connected=False, error=message)

    def _tick(self) -> None:
        scheduled = self.next_tick_at
        frame, positions, observed = self.devices.observe()
        sent = self.devices.command()
        commanded = time.monotonic()
        if self.writer is not None:
            self.writer.append(frame.rgb, frame.sequence, frame.timestamp, observed, commanded, positions, sent,
                               self.devices.secondary_frame())
        finished = time.monotonic()
        duration = finished - scheduled
        age = observed - frame.timestamp
        if self.writer is not None:
            self.writer.record_timing(duration, age)
            self._update(recorded_rows=len(self.writer.rows))
        with self.lock:
            diagnostics = self.state['diagnostics']
            diagnostics['iterations'] += 1
            diagnostics['missed_deadlines'] += int(duration > 1 / 30)
            diagnostics['max_cycle_s'] = max(diagnostics['max_cycle_s'], duration)
            diagnostics['camera_age_s'] = age
        self.next_tick_at = max(scheduled + 1 / 30, finished)
        self.stop_event.wait(max(0.0, self.next_tick_at - time.monotonic()))

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                try:
                    command, payload, future = self.commands.get(timeout=0 if self.devices else 0.1)
                except Empty:
                    pass
                else:
                    try:
                        self._command(command, payload)
                        if command == 'stop' and self.finalization is not None:
                            self.stop_response = future
                        else:
                            future.set_result(self.snapshot())
                    except Exception as error:
                        if command in ('start', 'stop', 'disconnect') and not isinstance(error, ValueError):
                            self._fault(error)
                        future.set_exception(error)
                self._complete_finalization()
                if self.devices is not None:
                    try:
                        self._tick()
                    except Exception as error:
                        self._fault(error)
        finally:
            try:
                if self.writer is not None:
                    self._begin_finalization('Recorder server shut down during recording.')
            finally:
                self._disconnect()
