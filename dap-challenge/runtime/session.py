"""Single-owner live execution shared by policy evaluation and future rollout collectors."""

from concurrent.futures import Future
from copy import deepcopy
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
import math
import time
from typing import Any, Callable

from runtime.policy import PolicyRunner
from shared.config import load_config, save_config
from shared.devices import CameraFrame, FollowerDevices
from shared.normalization import JOINT_NAMES, normalize_controller
from shared.ownership import HardwareLease
import torch


class RuntimeSession:
    def __init__(self, settings: dict, pose_path: Path, lease: HardwareLease,
                 factory: Callable = FollowerDevices, clock: Callable = time.monotonic):
        control = settings['control']
        if any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0 for value in control.values()):
            raise ValueError('Runtime control settings must be positive and finite.')
        if not isinstance(control['consecutive_deadline_misses'], int):
            raise ValueError('Deadline miss count must be an integer.')

        self.settings, self.pose_path, self.lease = deepcopy(settings), pose_path, lease
        self.factory, self.clock = factory, clock
        self.devices = None
        self.runner: PolicyRunner | None = None
        self.metadata = {}
        self.pose = load_config(pose_path) if pose_path.exists() else None
        self.client_id = None
        self.heartbeat_at = 0.0
        self.motion_cancel = Event()
        self.shutdown = Event()
        self.commands = Queue()
        self.lock = Lock()
        self.frames: tuple[CameraFrame, CameraFrame] | None = None
        self.state = {'connected': False, 'phase': 'disconnected', 'error': None, 'hold_confirmed': False,
                      'policy_version': None, 'action_mode': settings['policy']['action_mode'], 'starting_pose': self.pose,
                      'positions': None, 'requested_target': None, 'sent_target': None,
                      'diagnostics': {'iterations': 0, 'missed_deadlines': 0, 'consecutive_misses': 0,
                                      'cycle_ms': 0.0, 'prediction_ms': 0.0, 'camera_age_s': 0.0, 'limited_commands': 0}}
        self.thread = Thread(target=self._run, name='policy-runtime', daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.motion_cancel.set()
        self.shutdown.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise TimeoutError('Runtime worker did not stop; do not start another hardware owner.')

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {**deepcopy(self.state), 'settings': deepcopy(self.settings)}

    def _update(self, **values: Any) -> None:
        with self.lock:
            self.state.update(values)

    def preview_identity(self, client_id: str) -> object:
        with self.lock:
            if self.client_id != client_id or not self.state['connected'] or self.devices is None:
                raise ValueError('Connect this browser before starting camera previews.')

            return self.devices

    def preview_frame(self, view: str, client_id: str, identity: object) -> CameraFrame:
        with self.lock:
            if self.client_id != client_id or self.devices is not identity or not self.state['connected'] or self.frames is None:
                raise ValueError('Camera session is no longer available.')

            return self.frames[0 if view == 'arm' else 1]

    def heartbeat(self, client_id: str) -> None:
        with self.lock:
            if self.client_id != client_id or not self.state['connected']:
                raise ValueError('This browser does not own the inference session.')

            self.heartbeat_at = self.clock()

    def request(self, command: str, payload: dict | None = None) -> dict:
        if not self.thread.is_alive() or self.shutdown.is_set():
            raise RuntimeError('Inference worker is not running.')
        values = payload or {}
        if command in ('stop', 'disconnect') and values.get('client_id') == self.client_id:
            self.motion_cancel.set()

        future = Future()
        self.commands.put((command, values, future))
        try:
            return future.result(timeout=30)
        except TimeoutError:
            future.cancel()
            raise

    def publish(self, runner: PolicyRunner) -> dict:
        """Adopt a caller-owned snapshot only between attempts; never mutate active weights."""
        return self.request('publish', {'runner': runner})

    def _signature(self) -> dict:
        return deepcopy({'follower': self.metadata['follower'], 'calibration': self.metadata['calibration']['follower'],
                         'joint_names': list(JOINT_NAMES)})

    def _hold(self) -> None:
        self._update(hold_confirmed=False)
        positions = self.devices.read_positions()
        normalize_controller(torch.tensor(positions, dtype=torch.float32), JOINT_NAMES)
        sent = self.devices.send_target(positions)
        self._update(positions=positions, sent_target=sent, hold_confirmed=True)
        if self.runner is not None:
            self.runner.reset()

    def _fault(self, error: Exception) -> None:
        self.motion_cancel.set()
        message = str(error)
        try:
            self._hold()
        except Exception as failure:
            message += f'; HOLD UNCONFIRMED: {failure}'
        self._update(phase='fault', error=message)

    def _disconnect(self) -> None:
        if self.devices is not None:
            self.devices.disconnect()
            self.devices = None
        self.lease.release('policy runtime')
        self.client_id = None
        self.frames = None
        if self.runner is not None:
            self.runner.reset()
        self._update(connected=False, phase='disconnected', hold_confirmed=False, positions=None)

    def _command(self, command: str, payload: dict) -> None:
        if command in ('load', 'publish'):
            if self.state['phase'] in ('running', 'resetting'):
                raise ValueError('Stop the policy before changing its snapshot.')
            if self.devices is not None and command == 'load' and payload.get('client_id') != self.client_id:
                raise ValueError('This browser does not own the inference session.')

            runner = payload['runner']
            self.runner = runner
            runner.reset()
            self._update(policy_version=runner.version)
            return
        if command == 'connect':
            if self.devices is not None:
                raise ValueError('Disconnect the current session first.')
            if not payload.get('client_id'):
                raise ValueError('A browser owner ID is required.')

            self.lease.acquire('policy runtime')
            try:
                self.devices = self.factory(payload['settings'])
                self.metadata = self.devices.connect()
                self._hold()
                self.client_id = payload['client_id']
                self.heartbeat_at = self.clock()
                self._update(connected=True, phase='holding', error=None)
            except Exception:
                self._disconnect()
                raise
            return
        if self.devices is None or payload.get('client_id') != self.client_id:
            raise ValueError('Connect this browser to the follower first.')

        if command == 'disconnect':
            self._disconnect()
        elif command == 'stop':
            self._hold()
            self._update(phase='holding', error=None)
        elif command == 'capture_pose':
            if self.state['phase'] in ('running', 'resetting'):
                raise ValueError('Stop motion before capturing the starting pose.')

            self._hold()
            pose = {'positions': self.state['positions'], 'identity': self._signature()}
            save_config(pose, self.pose_path)
            self.pose = pose
            self._update(starting_pose=pose)
        elif command in ('start', 'reset'):
            if self.state['phase'] in ('running', 'resetting'):
                raise ValueError('Stop the current attempt first.')
            if self.pose is None or self.pose['identity'] != self._signature():
                raise ValueError('Capture a starting pose for this follower and calibration first.')
            if self.clock() - self.heartbeat_at > self.settings['control']['browser_timeout_s']:
                raise ValueError('Browser heartbeat expired.')

            positions = self.devices.read_positions()
            normalize_controller(torch.tensor(self.pose['positions'], dtype=torch.float32), JOINT_NAMES)
            if command == 'start':
                if self.runner is None or payload.get('mode') not in ('mean', 'sample'):
                    raise ValueError('Load a checkpoint and select mean or sample mode first.')
                if max(abs(a-b) for a,b in zip(positions, self.pose['positions'], strict=True)) > self.settings['control']['start_tolerance']:
                    raise ValueError('Return to the captured starting pose before starting the policy.')

                self.runner.reset()
                self._update(phase='running', action_mode=payload['mode'], error=None, hold_confirmed=False)
            else:
                self.reset_started = self.clock()
                self._update(phase='resetting', error=None, hold_confirmed=False)
            self.state['diagnostics']['consecutive_misses'] = 0
            self.motion_cancel.clear()
        else:
            raise ValueError('Unknown inference command.')

    def _tick(self, started: float, elapsed: float) -> None:
        arm, positions, observed = self.devices.observe()
        mac = self.devices.secondary_frame()
        if mac is None:
            raise ValueError('Live arm and Mac cameras are both required.')

        now = self.clock()
        age = max(now-arm.timestamp, now-mac.timestamp)
        with self.lock:
            self.frames = (arm, mac)
            self.state['positions'] = positions
            self.state['diagnostics']['camera_age_s'] = age
        active = self.state['phase'] in ('running', 'resetting')
        if active and age > self.settings['control']['camera_stale_s']:
            raise TimeoutError('Live camera frame is stale; policy stopped.')
        if active and now-self.heartbeat_at > self.settings['control']['browser_timeout_s']:
            raise TimeoutError('Browser heartbeat lost; policy stopped.')
        if not active or self.motion_cancel.is_set():
            return

        normalize_controller(torch.tensor(positions, dtype=torch.float32), JOINT_NAMES)
        control = self.settings['control']
        if self.state['phase'] == 'resetting':
            if now-self.reset_started > control['reset_timeout_s']:
                raise TimeoutError('Return to start timed out.')
            if max(abs(a-b) for a,b in zip(positions,self.pose['positions'],strict=True)) <= control['reset_tolerance']:
                self._hold()
                self._update(phase='holding')
                return

            delta = min(control['max_target_delta'], control['reset_speed_per_second'] * min(elapsed, .1))
            requested = self.pose['positions']
            targets = [max(p-delta,min(p+delta,t)) for p,t in zip(positions,requested,strict=True)]
        else:
            self.runner.observe(arm, mac, positions, self.metadata['camera']['rotation_clockwise'])
            prediction_started = self.clock()
            decision = self.runner.predict(self.state['action_mode'])
            self.state['diagnostics']['prediction_ms'] = (self.clock()-prediction_started)*1000
            requested = decision.requested
            delta = control['max_target_delta']
            targets = [max(p-delta,min(p+delta,t)) for p,t in zip(positions,requested,strict=True)]
        normalize_controller(torch.tensor(requested,dtype=torch.float32), JOINT_NAMES)
        normalize_controller(torch.tensor(targets,dtype=torch.float32), JOINT_NAMES)
        if self.motion_cancel.is_set():
            return
        if max(self.clock()-arm.timestamp, self.clock()-mac.timestamp) > control['camera_stale_s'] or self.clock()-self.heartbeat_at > control['browser_timeout_s']:
            raise TimeoutError('Prediction became stale before command transmission.')

        sent = self.devices.send_target(targets)
        self._update(requested_target=requested, sent_target=sent)
        if targets != requested:
            self.state['diagnostics']['limited_commands'] += 1

    def _run(self) -> None:
        period = 1 / self.settings['control']['iterations_per_second']
        previous = self.clock()-period
        try:
            while not self.shutdown.is_set():
                started = self.clock()
                try:
                    command,payload,future = self.commands.get(timeout=0 if self.devices else .1)
                except Empty:
                    pass
                else:
                    if future.set_running_or_notify_cancel():
                        try:
                            self._command(command,payload)
                            future.set_result(self.snapshot())
                        except Exception as error:
                            if self.devices is not None and (command in ('stop','disconnect','capture_pose') or not isinstance(error,ValueError)):
                                self._fault(error)
                            future.set_exception(error)
                if self.devices is not None:
                    active = self.state['phase'] in ('running','resetting')
                    try:
                        self._tick(started, self.clock()-previous)
                        duration = self.clock()-started
                        with self.lock:
                            diagnostics = self.state['diagnostics']
                            diagnostics['iterations'] += 1
                            diagnostics['cycle_ms'] = duration*1000
                            diagnostics['missed_deadlines'] += int(duration>period)
                            diagnostics['consecutive_misses'] = diagnostics['consecutive_misses']+1 if active and duration>period else 0
                        if active and diagnostics['consecutive_misses'] >= self.settings['control']['consecutive_deadline_misses']:
                            raise TimeoutError('Repeated control deadlines missed; policy stopped.')
                    except Exception as error:
                        if self.state['phase'] != 'fault':
                            self._fault(error)
                previous = started
                self.shutdown.wait(max(0, period-(self.clock()-started)))
        finally:
            try:
                self._disconnect()
            except Exception as error:
                self._update(phase='fault', error=f'Disconnect unconfirmed: {error}')
