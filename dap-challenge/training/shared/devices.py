"""Single-owner SO-101 I/O with calibrated ranges and host-timestamped RGB frames."""

from dataclasses import asdict, dataclass
from importlib.metadata import version
import math
from pathlib import Path
from threading import Event, Lock, Thread
import sys
import time
from typing import Any, Iterator, Protocol

import av
from av.container import InputContainer
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


def open_camera(
    index: int, width: int | None = None, height: int | None = None, fps: int | None = None,
) -> InputContainer:
    """Open a macOS video device through PyAV without opening an audio device."""
    if sys.platform != 'darwin':
        raise RuntimeError('Camera capture currently supports macOS AVFoundation only.')
    if index < 0:
        raise ValueError('camera index must be nonnegative')
    if (width is None) != (height is None):
        raise ValueError('specify both capture width and height')

    options = {'pixel_format': 'nv12'}
    if width is not None:
        options['video_size'] = f'{width}x{height}'
    if fps is not None:
        options['framerate'] = str(fps)
    return av.open(f'{index}:none', format='avfoundation', options=options, timeout=8)


def camera_frames(capture: InputContainer, stop: Event) -> Iterator[av.VideoFrame]:
    """Retry temporarily empty live reads; fail on EOF or ten seconds without frames."""
    packets = capture.demux(video=0)
    last_frame = time.monotonic()
    while not stop.is_set():
        try:
            packet = next(packets)
            for frame in packet.decode():
                last_frame = time.monotonic()
                yield frame
        except av.error.BlockingIOError:
            if time.monotonic() - last_frame >= 10:
                raise TimeoutError('camera supplied no frame for ten seconds')

            stop.wait(0.005)
            packets = capture.demux(video=0)
        except StopIteration as error:
            raise ConnectionError('camera stream ended') from error


def reported_camera_rate(capture: InputContainer) -> float | None:
    """Represent unavailable or non-finite device frame rates as unknown metadata."""
    rate = capture.streams.video[0].average_rate
    if rate is None:
        return None

    value = float(rate)
    if not math.isfinite(value) or value <= 0:
        return None

    return value


class CameraCapture:
    def __init__(self, index: int, width: int, height: int, fps: int, rotation_clockwise: int = 0):
        if rotation_clockwise not in (0, 90, 180, 270):
            raise ValueError('camera rotation must be 0, 90, 180 or 270 degrees')

        self.rotation_clockwise = rotation_clockwise
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self.lock = Lock()
        self.stop = Event()
        self.ready = Event()
        self.latest: CameraFrame | None = None
        self.error: str | None = None
        self.reported_fps = None
        self.source_pixel_format = None
        self.thread = Thread(target=self._capture, daemon=True)

    def connect(self) -> None:
        self.thread.start()
        if not self.ready.wait(10):
            self.stop.set()
            raise TimeoutError('camera did not supply a frame within 10 seconds')

        self.read()

    def _capture(self) -> None:
        try:
            with open_camera(self.index, self.width, self.height, self.fps) as capture:
                self.reported_fps = reported_camera_rate(capture)
                for sequence, frame in enumerate(camera_frames(capture, self.stop)):
                    timestamp = time.monotonic()
                    if (frame.height, frame.width) != (self.height, self.width):
                        raise ValueError('camera supplied different dimensions from the selected capture mode')

                    rgb: UInt8[np.ndarray, 'H_src W_src C_in'] = frame.to_ndarray(format='rgb24')
                    rgb = np.ascontiguousarray(np.rot90(rgb, k=-(self.rotation_clockwise // 90)))
                    self.source_pixel_format = frame.format.name
                    with self.lock:
                        self.latest = CameraFrame(rgb, sequence, timestamp)
                    self.ready.set()
        except Exception as error:
            with self.lock:
                self.error = str(error)
            self.ready.set()

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
            self.camera = CameraCapture(config['camera_index'], config['width'], config['height'], config['camera_fps'],
                                        config.get('rotation_clockwise', 90))
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
                'camera': {'index': config['camera_index'], 'width': self.camera.read().rgb.shape[1], 'height': self.camera.read().rgb.shape[0],
                           'capture_width': config['width'], 'capture_height': config['height'],
                           'rotation_clockwise': self.camera.rotation_clockwise,
                           'requested_frames_per_second': config['camera_fps'],
                           'reported_frames_per_second': self.camera.reported_fps,
                           'backend': 'pyav_avfoundation', 'source_pixel_format': self.camera.source_pixel_format,
                           'color': 'RGB',
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
    """Probe native frame geometry at 30 images per second, not every supported mode."""
    with open_camera(index, fps=30) as capture:
        frame = next(camera_frames(capture, Event()))
        rate = reported_camera_rate(capture)
        return {'index': index, 'width': frame.width, 'height': frame.height,
                'reported_frames_per_second': rate, 'requested_frames_per_second': 30,
                'backend': 'pyav_avfoundation', 'source_pixel_format': frame.format.name,
                'note': 'Profile probed at 30 images per second; requested dimensions are checked when connecting.'}
