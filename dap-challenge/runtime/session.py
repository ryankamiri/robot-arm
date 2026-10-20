"""Single-owner live execution shared by policy evaluation and scored RL collection."""

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
        self.pending_runner: PolicyRunner | None = None
        self.publications: dict[str, tuple[int, str]] = {}
        self.metadata = {}
        self.rl_owner = None
        self.rl_run = None
        self.rollout_store = None
        self.collector = None
        self.last_collector = None
        self.pose = load_config(pose_path) if pose_path.exists() else None
        self.client_id = None
        self.heartbeat_at = 0.0
        self.motion_cancel = Event()
        self.shutdown = Event()
        self.commands = Queue()
        self.lock = Lock()
        self.frames: tuple[CameraFrame, CameraFrame] | None = None
        self.state = {'connected': False, 'phase': 'disconnected', 'error': None, 'hold_confirmed': False,
                      'policy_version': None, 'pending_policy_version': None, 'action_mode': settings['policy']['action_mode'], 'starting_pose': self.pose,
                      'positions': None, 'requested_target': None, 'sent_target': None,
                      'diagnostics': {'iterations': 0, 'missed_deadlines': 0, 'consecutive_misses': 0,
                                      'cycle_ms': 0.0, 'prediction_ms': 0.0, 'camera_age_s': 0.0, 'limited_commands': 0}}
        self.thread = Thread(target=self.run_loop, name='policy-runtime', daemon=True)

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

    def update_state(self, **values: Any) -> None:
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
        if command in ('stop', 'abort', 'disconnect', 'reset', 'rl_release') and values.get('client_id') == self.client_id:
            self.motion_cancel.set()

        future = Future()
        self.commands.put((command, values, future))
        try:
            return future.result(timeout=30)
        except TimeoutError:
            future.cancel()
            raise

    def publish(self, runner: PolicyRunner) -> dict:
        """Stage a stable snapshot during an attempt or adopt it while idle."""
        return self.request('publish', {'runner': runner})

    def adopt_policy(self, runner: PolicyRunner) -> None:
        if isinstance(runner, PolicyRunner):
            runner.to_device(self.settings['policy']['device'])
        runner.reset()
        self.runner = runner
        self.pending_runner = None
        self.update_state(policy_version=runner.version, pending_policy_version=None)

    def stage_publication(self, runner: PolicyRunner) -> None:
        publication = getattr(runner, 'publication', None)
        if self.rl_run and (publication is None or publication.run_id != self.rl_run):
            raise ValueError('Only the active RL learner may publish during this session.')

        if publication is not None:
            previous = self.publications.get(publication.run_id)
            if previous is not None:
                step, digest = previous
                if publication.actor_updates < step:
                    raise ValueError('Refusing an older actor publication.')

                if publication.actor_updates == step:
                    if publication.digest != digest:
                        raise ValueError('Publication version already identifies different weights.')

                    return
        if self.state['phase'] in ('running', 'resetting', 'fault'):
            self.pending_runner = runner
            self.update_state(pending_policy_version=runner.version)
        else:
            self.adopt_policy(runner)
        if publication is not None:
            self.publications[publication.run_id] = (publication.actor_updates, publication.digest)

    def follower_signature(self) -> dict:
        return deepcopy({'follower': self.metadata['follower'], 'calibration': self.metadata['calibration']['follower'],
                         'joint_names': list(JOINT_NAMES)})

    def hold_position(self) -> None:
        self.update_state(hold_confirmed=False)
        positions = self.devices.read_positions()
        normalize_controller(torch.tensor(positions, dtype=torch.float32), JOINT_NAMES)
        sent = self.devices.send_target(positions)
        self.update_state(positions=positions, sent_target=sent, hold_confirmed=True)
        if self.runner is not None:
            self.runner.reset()

    def handle_fault(self, error: Exception) -> None:
        self.motion_cancel.set()
        self.abort_attempt(str(error))
        message = str(error)
        if self.state['connected']:
            try:
                self.hold_position()
            except Exception as failure:
                message += f'; HOLD UNCONFIRMED: {failure}'
        self.update_state(phase='fault', error=message)

    def abort_attempt(self, reason: str) -> None:
        if self.collector is not None:
            self.collector.finish(error=reason)
            self.last_collector = self.collector
            self.collector = None

    def disconnect(self) -> None:
        self.abort_attempt('Disconnected before this attempt was scored.')
        if self.devices is not None:
            self.devices.disconnect()
            self.devices = None
        self.lease.release('policy runtime')
        self.client_id = None
        self.frames = None
        if self.runner is not None:
            self.runner.reset()
        self.update_state(connected=False, phase='disconnected', error=None, hold_confirmed=False, positions=None)

    def handle_command(self, command: str, payload: dict) -> None:
        if command == 'rl_reserve':
            if self.state['phase'] in ('running', 'resetting') or (self.client_id and self.client_id != payload['client_id']):
                raise ValueError('Stop motion and use the browser that owns the follower before loading a run.')

            self.adopt_policy(payload['runner'])
            self.rl_owner, self.rl_run = payload['client_id'], payload['run_id']
            self.rollout_store = payload['store']
            return
        if command == 'rl_release':
            if self.rl_owner != payload.get('client_id'):
                raise ValueError('This browser does not own the RL session.')

            self.disconnect()
            self.rl_owner = self.rl_run = self.rollout_store = None
            return
        if self.rl_owner is not None:
            if command == 'load' or (command in ('connect', 'start') and payload.get('client_id') != self.rl_owner):
                raise ValueError('The RL session owns this runtime.')
            if command == 'start' and not payload.get('rl_attempt'):
                raise ValueError('Start recorded attempts from the RL route.')
            if command == 'reset' and self.collector is not None:
                raise ValueError('Stop and review the RL attempt before returning to start.')

        if command == 'publish':
            self.stage_publication(payload['runner'])
            return
        if command == 'load':
            if self.state['phase'] in ('running', 'resetting'):
                raise ValueError('Stop the policy before changing its snapshot.')

            if self.devices is not None and payload.get('client_id') != self.client_id:
                raise ValueError('This browser does not own the inference session.')

            self.adopt_policy(payload['runner'])
            return
        if command == 'connect':
            if self.devices is not None:
                if self.state['connected']:
                    raise ValueError('Disconnect the current session first.')

                try:
                    self.disconnect()
                except Exception as error:
                    raise RuntimeError(f'Previous device cleanup is incomplete: {error}') from error

            if not payload.get('client_id'):
                raise ValueError('A browser owner ID is required.')

            self.lease.acquire('policy runtime')
            try:
                self.devices = self.factory(payload['settings'])
                self.metadata = self.devices.connect()
                self.hold_position()
                self.client_id = payload['client_id']
                self.heartbeat_at = self.clock()
                self.update_state(connected=True, phase='holding', error=None)
            except Exception:
                self.disconnect()
                raise
            return
        if self.devices is None or payload.get('client_id') != self.client_id:
            raise ValueError('Connect this browser to the follower first.')

        if command == 'disconnect':
            self.disconnect()
        elif command == 'abort':
            self.abort_attempt('Attempt aborted by browser.')
            self.hold_position()
            self.update_state(phase='holding', error=None)
        elif command == 'stop':
            if self.collector is not None:
                arm, positions, observed = self.devices.observe()
                mac = self.devices.secondary_frame()
                if mac is None or max(self.clock()-arm.timestamp, self.clock()-mac.timestamp) > self.settings['control']['camera_stale_s']:
                    raise ValueError('Final camera observation is stale or missing.')

                collector = self.collector
                self.hold_position()
                collector.finish((arm, mac, positions, observed))
                self.last_collector, self.collector = collector, None
            else:
                self.hold_position()
            self.update_state(phase='holding', error=None)
        elif command == 'capture_pose':
            if self.state['phase'] in ('running', 'resetting'):
                raise ValueError('Stop motion before capturing the starting pose.')

            self.hold_position()
            pose = {'positions': self.state['positions'], 'identity': self.follower_signature()}
            save_config(pose, self.pose_path)
            self.pose = pose
            self.update_state(starting_pose=pose)
        elif command in ('start', 'reset'):
            if self.state['phase'] == 'resetting' or (command == 'start' and self.state['phase'] == 'running'):
                raise ValueError('Stop the current attempt first.')

            if self.pose is None or self.pose['identity'] != self.follower_signature():
                raise ValueError('Capture a starting pose for this follower and calibration first.')

            if self.clock() - self.heartbeat_at > self.settings['control']['browser_timeout_s']:
                raise ValueError('Browser heartbeat expired.')

            positions = self.devices.read_positions()
            normalize_controller(torch.tensor(self.pose['positions'], dtype=torch.float32), JOINT_NAMES)
            if command == 'start':
                if (self.runner is None and self.pending_runner is None) or payload.get('mode') not in ('mean', 'sample'):
                    raise ValueError('Load a checkpoint and select mean or sample mode first.')

                if max(abs(a-b) for a,b in zip(positions, self.pose['positions'], strict=True)) > self.settings['control']['start_tolerance']:
                    raise ValueError('Return to the captured starting pose before starting the policy.')

                if self.pending_runner is not None:
                    self.adopt_policy(self.pending_runner)
                self.runner.reset()
                if payload.get('rl_attempt'):
                    if self.rollout_store is None or self.collector is not None or (self.last_collector and self.last_collector.thread.is_alive()):
                        raise ValueError('Wait for the previous RL attempt to finalize.')

                    self.collector = self.rollout_store.begin(payload['tasks'], self.runner.version, self.metadata)
                self.update_state(phase='running', action_mode=payload['mode'], error=None, hold_confirmed=False)
            else:
                if self.runner is not None:
                    self.runner.reset()
                self.reset_started = self.clock()
                self.reset_target = positions.copy()
                self.update_state(phase='resetting', error=None, hold_confirmed=False)
            self.state['diagnostics']['consecutive_misses'] = 0
            self.motion_cancel.clear()
        else:
            raise ValueError('Unknown inference command.')

    def tick(self, started: float, elapsed: float) -> None:
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
                self.hold_position()
                self.update_state(phase='holding')
                return

            delta = min(control['max_target_delta'], control['reset_speed_per_second'] * min(elapsed, .1))
            requested = self.pose['positions']
            # Advance the command trajectory even when a motor has not reached the previous small goal.
            planned = [max(p-delta,min(p+delta,t)) for p,t in zip(self.reset_target,requested,strict=True)]
            limit = control['max_target_delta']
            targets = [max(p-limit,min(p+limit,t)) for p,t in zip(positions,planned,strict=True)]
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
        if self.collector is not None and self.state['phase'] == 'running':
            self.collector.transition(arm, mac, positions, observed, requested, sent, self.clock())
        if self.state['phase'] == 'resetting':
            self.reset_target = list(sent)
        self.update_state(requested_target=requested, sent_target=sent)
        if targets != requested:
            self.state['diagnostics']['limited_commands'] += 1

    def run_loop(self) -> None:
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
                            self.handle_command(command,payload)
                            future.set_result(self.snapshot())
                        except Exception as error:
                            if self.devices is not None and (command in ('stop','abort','disconnect','capture_pose','reset','rl_release') or not isinstance(error,ValueError)):
                                self.handle_fault(error)
                            future.set_exception(error)
                if self.devices is not None and self.state['connected']:
                    active = self.state['phase'] in ('running','resetting')
                    try:
                        self.tick(started, self.clock()-previous)
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
                            self.handle_fault(error)
                previous = started
                self.shutdown.wait(max(0, period-(self.clock()-started)))
        finally:
            try:
                self.disconnect()
            except Exception as error:
                self.update_state(phase='fault', error=f'Disconnect unconfirmed: {error}')
