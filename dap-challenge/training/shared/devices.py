"""Single-owner SO-101 I/O with calibrated ranges and host-timestamped RGB frames."""

from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from threading import Event, Lock, Thread
import time
from typing import Any, Protocol

import cv2
from jaxtyping import UInt8
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
import numpy as np

from training.shared.normalization import JOINT_NAMES


@dataclass
class CameraFrame:
    rgb: UInt8[np.ndarray, 'H_src W_src C_in']
    sequence: int
    timestamp: float


class Devices(Protocol):
    def connect(self) -> dict[str, Any]: ...
    def observe(self) -> tuple[CameraFrame, list[float], float]: ...
    def command(self) -> list[float]: ...
    def disconnect(self) -> None: ...


class CameraCapture:
    def __init__(self, index: int, width: int, height: int, fps: int):
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self.lock = Lock()
        self.stop = Event()
        self.ready = Event()
        self.latest: CameraFrame | None = None
        self.error: str | None = None
        self.reported_fps = None
        self.thread = Thread(target=self._capture, daemon=True)

    def connect(self) -> None:
        self.thread.start()
        if not self.ready.wait(10):
            raise TimeoutError('camera did not supply a frame within 10 seconds')

        self.read()

    def _capture(self) -> None:
        capture = cv2.VideoCapture(self.index)
        try:
            if not capture.isOpened():
                raise ConnectionError('camera could not be opened')

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            capture.set(cv2.CAP_PROP_FPS, self.fps)
            actual = (round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            if actual != (self.width, self.height):
                raise ValueError(f'camera supplied {actual}, not requested {(self.width, self.height)}')

            self.reported_fps = capture.get(cv2.CAP_PROP_FPS)
            sequence = 0
            while not self.stop.is_set():
                ok, bgr = capture.read()
                timestamp = time.monotonic()
                if not ok:
                    raise ConnectionError('camera stopped supplying frames')
                if bgr.shape[:2] != (self.height, self.width):
                    raise ValueError('camera frame dimensions changed from the selected capture mode')

                rgb: UInt8[np.ndarray, 'H_src W_src C_in'] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.latest = CameraFrame(rgb, sequence, timestamp)
                sequence += 1
                self.ready.set()
        except Exception as error:
            with self.lock:
                self.error = str(error)
            self.ready.set()
        finally:
            capture.release()

    def read(self) -> CameraFrame:
        with self.lock:
            if self.error:
                raise ConnectionError(self.error)
            if self.latest is None:
                raise ConnectionError('no camera frame is available')

            return self.latest

    def disconnect(self) -> None:
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise TimeoutError('camera capture is blocked; restart the backend before reconnecting')


class SO101Devices:
    def __init__(self, settings: dict[str, Any]):
        self.settings = settings
        self.camera = None
        self.leader = None
        self.follower = None

    def connect(self) -> dict[str, Any]:
        config = self.settings
        common = {'use_degrees': False}
        if config.get('calibration_dir'):
            common['calibration_dir'] = Path(config['calibration_dir']).expanduser()
        self.leader = SO101Leader(SO101LeaderConfig(port=config['leader_port'], id=config['leader_id'], **common))
        self.follower = SO101Follower(SO101FollowerConfig(
            port=config['follower_port'], id=config['follower_id'],
            disable_torque_on_disconnect=True, cameras={}, **common,
        ))
        if not self.leader.calibration or not self.follower.calibration:
            raise ValueError('Calibrate both arm IDs with LeRobot first; web connections never prompt for calibration.')

        try:
            self.camera = CameraCapture(config['camera_index'], config['width'], config['height'], config['camera_fps'])
            self.camera.connect()
            self.leader.connect(calibrate=False)
            if not self.leader.is_calibrated:
                raise ValueError('leader calibration does not match the connected motors')

            self.follower.connect(calibrate=False)
            if not self.follower.is_calibrated:
                raise ValueError('follower calibration does not match the connected motors')

            return {
                'device_mode': 'physical', 'lerobot_version': version('lerobot'),
                'leader': {'port': config['leader_port'], 'id': config['leader_id']},
                'follower': {'port': config['follower_port'], 'id': config['follower_id']},
                'calibration': {
                    'leader': {name: asdict(value) for name, value in self.leader.calibration.items()},
                    'follower': {name: asdict(value) for name, value in self.follower.calibration.items()},
                },
                'camera': {'index': config['camera_index'], 'width': config['width'], 'height': config['height'],
                           'requested_frames_per_second': config['camera_fps'],
                           'reported_frames_per_second': self.camera.reported_fps, 'color': 'RGB',
                           'timestamp_basis': 'host_read_complete_not_sensor_exposure'},
                'action_timestamp_basis': 'host_command_write_return_not_motor_arrival',
            }
        except Exception:
            self.disconnect()
            raise

    def observe(self) -> tuple[CameraFrame, list[float], float]:
        frame = self.camera.read()
        observation = self.follower.get_observation()
        positions = [float(observation[f'{name}.pos']) for name in JOINT_NAMES]
        return frame, positions, time.monotonic()

    def command(self) -> list[float]:
        requested = self.leader.get_action()
        sent = self.follower.send_action(requested)
        return [float(sent[f'{name}.pos']) for name in JOINT_NAMES]

    def disconnect(self) -> None:
        errors = []
        for arm in (self.follower, self.leader):
            try:
                if arm is not None and arm.bus.is_connected:
                    arm.disconnect()
            except Exception as error:
                errors.append(str(error))
        try:
            if self.camera is not None:
                self.camera.disconnect()
        except Exception as error:
            errors.append(str(error))
        if errors:
            raise RuntimeError('Device cleanup failed: ' + '; '.join(errors))


class DemoDevices:
    """Explicit synthetic UI/test source; never opens a camera or a serial port."""

    def connect(self) -> dict[str, Any]:
        self.sequence = 0
        self.positions = [0.0] * 5 + [50.0]
        return {'device_mode': 'demo', 'calibration': {'source': 'synthetic; not hardware calibration'},
                'camera': {'width': 320, 'height': 240, 'timestamp_basis': 'synthetic_host_monotonic'},
                'action_timestamp_basis': 'synthetic_command_return'}

    def observe(self) -> tuple[CameraFrame, list[float], float]:
        self.sequence += 1
        rgb: UInt8[np.ndarray, 'H_src W_src C_in'] = np.zeros((240, 320, 3), dtype=np.uint8)
        rgb[:, :, 1] = 40
        rgb[:, self.sequence % 280:self.sequence % 280 + 40, 0] = 200
        timestamp = time.monotonic()
        return CameraFrame(rgb, self.sequence, timestamp), self.positions.copy(), timestamp

    def command(self) -> list[float]:
        self.positions[0] = float(20 * np.sin(self.sequence / 20))
        return self.positions.copy()

    def disconnect(self) -> None:
        pass


def inspect_camera(index: int) -> dict[str, Any]:
    """Read the selected camera's reported default profile, not an exhaustive mode list."""
    capture = cv2.VideoCapture(index)
    try:
        if not capture.isOpened():
            raise ConnectionError('camera could not be opened')

        return {'index': index, 'width': round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'reported_frames_per_second': capture.get(cv2.CAP_PROP_FPS),
                'note': 'Reported default profile; requested modes are checked when connecting.'}
    finally:
        capture.release()
