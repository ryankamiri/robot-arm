"""Synthetic devices shared by explicit demo mode and automated tests."""

import time
from typing import Any

from jaxtyping import UInt8
import numpy as np

from shared.devices import CameraFrame
from shared.normalization import JOINT_NAMES


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

    def secondary_frame(self) -> CameraFrame | None:
        return None

    def command(self) -> list[float]:
        self.positions[0] = float(20 * np.sin(self.sequence / 20))
        return self.positions.copy()

    def disconnect(self) -> None:
        pass


class DemoFollower(DemoDevices):
    def connect(self) -> dict:
        metadata = super().connect()
        self.positions = [0.] * 5 + [50.]
        metadata.update(follower={'id':'demo_follower','port':'synthetic'},
                        calibration={'follower':{'source':'synthetic'}}, mac_camera={'rotation_clockwise':0})
        metadata['camera']['rotation_clockwise'] = 0
        return metadata

    def observe(self):
        frame, _, timestamp = super().observe()
        self.mac_frame = frame
        return frame, self.positions.copy(), timestamp

    def secondary_frame(self):
        return self.mac_frame

    def read_positions(self) -> list[float]:
        return self.positions.copy()

    def send_target(self, targets: list[float]) -> list[float]:
        if len(targets) != len(JOINT_NAMES):
            raise ValueError('Expected six targets.')

        self.positions = list(targets)
        return self.positions.copy()
