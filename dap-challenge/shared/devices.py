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
from lerobot.motors.feetech import OperatingMode
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
import numpy as np

from shared.normalization import JOINT_NAMES


@dataclass
class CameraFrame:
    rgb: UInt8[np.ndarray, 'H_src W_src C_in']
    sequence: int
    timestamp: float


class Devices(Protocol):
    def connect(self) -> dict[str, Any]: ...
    def observe(self) -> tuple[CameraFrame, list[float], float]: ...
    def secondary_frame(self) -> CameraFrame | None: ...
    def command(self) -> list[float]: ...
    def disconnect(self) -> None: ...


def open_camera(
    index: int | str, width: int | None = None, height: int | None = None, fps: int | None = None,
) -> InputContainer:
    """Open a macOS video device through PyAV without opening an audio device."""
    if sys.platform != 'darwin':
        raise RuntimeError('Camera capture currently supports macOS AVFoundation only.')
    if isinstance(index, int) and index < 0:
        raise ValueError('camera index must be nonnegative')
    if isinstance(index, str) and (not index or ':' in index):
        raise ValueError('camera name must be nonempty and contain no colon')
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
    def __init__(self, index: int | str, width: int, height: int, fps: int, rotation_clockwise: int = 0):
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
        self.thread = Thread(target=self.capture_loop, daemon=True)

    def connect(self) -> None:
        self.thread.start()
        if not self.ready.wait(10):
            self.stop.set()
            raise TimeoutError('camera did not supply a frame within 10 seconds')

        self.read()

    def capture_loop(self) -> None:
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


class HoldingFollower(SO101Follower):
    """Initialize position goals from measured raw positions before enabling torque."""

    def configure(self) -> None:
        if not self.is_calibrated:
            raise ValueError('Follower calibration does not match; refusing torque enable.')

        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write('Operating_Mode', motor, OperatingMode.POSITION.value)
            for register, value in (
                ('P_Coefficient', self.config.position_p_coefficient),
                ('I_Coefficient', self.config.position_i_coefficient),
                ('D_Coefficient', self.config.position_d_coefficient),
            ):
                self.bus.write(register, motor, value)
            if motor == 'gripper':
                self.bus.write('Max_Torque_Limit', motor, 500)
                self.bus.write('Protection_Current', motor, 250)
                self.bus.write('Overload_Torque', motor, 25)
        positions = self.bus.sync_read('Present_Position', normalize=False, num_retry=self.config.num_read_retries)
        self.bus.sync_write('Goal_Position', positions, normalize=False)
        self.bus.enable_torque()


class SO101Devices:
    requires_leader = True
    follower_class = SO101Follower

    def __init__(self, settings: dict[str, Any]):
        self.settings = settings
        self.camera = None
        self.mac_camera = None
        self.mac_frame = None
        self.leader = None
        self.follower = None

    def connect(self) -> dict[str, Any]:
        config = self.settings
        common = {'use_degrees': False}
        if config.get('calibration_dir'):
            common['calibration_dir'] = Path(config['calibration_dir']).expanduser()
        if self.requires_leader:
            self.leader = SO101Leader(SO101LeaderConfig(port=config['leader_port'], id=config['leader_id'], **common))
        self.follower = self.follower_class(SO101FollowerConfig(
            port=config['follower_port'], id=config['follower_id'],
            disable_torque_on_disconnect=True, cameras={}, **common,
        ))
        if not self.follower.calibration or (self.leader is not None and not self.leader.calibration):
            raise ValueError('Calibrate required arm IDs first; web connections never prompt for calibration.')

        try:
            self.camera = CameraCapture(config.get('camera_name', config['camera_index']), config['width'], config['height'], config['camera_fps'],
                                        config.get('rotation_clockwise', 90))
            self.camera.connect()
            if config.get('mac_camera_index') is not None:
                if config['mac_camera_index'] == config['camera_index']:
                    raise ValueError('Arm and Mac cameras must be different devices.')

                self.mac_camera = CameraCapture(config.get('mac_camera_name', config['mac_camera_index']), config['width'], config['height'], config['camera_fps'])
                self.mac_camera.connect()
            if self.leader is not None:
                self.leader.connect(calibrate=False)
                if not self.leader.is_calibrated:
                    raise ValueError('leader calibration does not match the connected motors')

            self.follower.connect(calibrate=False)
            if not self.follower.is_calibrated:
                raise ValueError('follower calibration does not match the connected motors')

            return {
                'device_mode': 'physical', 'lerobot_version': version('lerobot'),
                'leader': ({'port': config['leader_port'], 'id': config['leader_id']} if self.leader else None),
                'follower': {'port': config['follower_port'], 'id': config['follower_id']},
                'calibration': {
                    'leader': ({name: asdict(value) for name, value in self.leader.calibration.items()} if self.leader else None),
                    'follower': {name: asdict(value) for name, value in self.follower.calibration.items()},
                },
                'camera': {'index': config['camera_index'], 'name': config.get('camera_name'), 'width': self.camera.read().rgb.shape[1], 'height': self.camera.read().rgb.shape[0],
                           'capture_width': config['width'], 'capture_height': config['height'],
                           'rotation_clockwise': self.camera.rotation_clockwise,
                           'requested_frames_per_second': config['camera_fps'],
                           'reported_frames_per_second': self.camera.reported_fps,
                           'backend': 'pyav_avfoundation', 'source_pixel_format': self.camera.source_pixel_format,
                           'color': 'RGB',
                           'timestamp_basis': 'host_read_complete_not_sensor_exposure'},
                'mac_camera': ({'index': config['mac_camera_index'], 'name': config.get('mac_camera_name'), 'width': config['width'],
                                'height': config['height'], 'rotation_clockwise': 0,
                                'requested_frames_per_second': config['camera_fps'],
                                'timestamp_basis': 'host_read_complete_not_sensor_exposure'} if self.mac_camera else None),
                'action_timestamp_basis': 'host_command_write_return_not_motor_arrival',
            }
        except Exception:
            self.disconnect()
            raise

    def observe(self) -> tuple[CameraFrame, list[float], float]:
        self.mac_frame = self.mac_camera.read() if self.mac_camera else None
        frame = self.camera.read()
        positions = self.read_positions()
        return frame, positions, time.monotonic()

    def read_positions(self) -> list[float]:
        observation = self.follower.get_observation()
        return [float(observation[f'{name}.pos']) for name in JOINT_NAMES]

    def send_target(self, targets: list[float]) -> list[float]:
        requested = {f'{name}.pos': value for name, value in zip(JOINT_NAMES, targets, strict=True)}
        sent = self.follower.send_action(requested)
        return [float(sent[f'{name}.pos']) for name in JOINT_NAMES]

    def secondary_frame(self) -> CameraFrame | None:
        return self.mac_frame

    def command(self) -> list[float]:
        requested = self.leader.get_action()
        return self.send_target([float(requested[f'{name}.pos']) for name in JOINT_NAMES])

    def disconnect(self) -> None:
        errors = []
        for arm in (self.follower, self.leader):
            try:
                if arm is not None and arm.bus.is_connected:
                    arm.disconnect()
            except Exception as error:
                errors.append(str(error))
        for camera in (self.camera, self.mac_camera):
            try:
                if camera is not None:
                    camera.disconnect()
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError('Device cleanup failed: ' + '; '.join(errors))


class FollowerDevices(SO101Devices):
    """Autonomous follower and two cameras; never opens or reads the leader."""

    requires_leader = False
    follower_class = HoldingFollower


def inspect_camera(index: int | str) -> dict[str, Any]:
    """Probe native frame geometry at 30 images per second, not every supported mode."""
    with open_camera(index, fps=30) as capture:
        frame = next(camera_frames(capture, Event()))
        rate = reported_camera_rate(capture)
        return {'index': index, 'width': frame.width, 'height': frame.height,
                'reported_frames_per_second': rate, 'requested_frames_per_second': 30,
                'backend': 'pyav_avfoundation', 'source_pixel_format': frame.format.name,
                'note': 'Profile probed at 30 images per second; requested dimensions are checked when connecting.'}
