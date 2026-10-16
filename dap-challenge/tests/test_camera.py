"""Verify live-read handling and RGB capture without loading duplicate FFmpeg builds."""

import errno
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import av
from jaxtyping import UInt8
import numpy as np

from training.shared.devices import CameraCapture, camera_frames, inspect_camera


class CameraTests(unittest.TestCase):
    def frame(self, width=8, height=6):
        rgb: UInt8[np.ndarray, 'H_src W_src C_in'] = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255
        return av.VideoFrame.from_ndarray(rgb, format='rgb24')

    def container(self):
        capture = MagicMock()
        capture.streams.video = [SimpleNamespace(average_rate=Fraction(30, 1))]
        return capture

    def test_temporary_empty_read_is_retried_without_ending_the_stream(self):
        capture = self.container()
        empty = MagicMock()
        empty.__next__.side_effect = av.error.BlockingIOError(errno.EAGAIN, 'temporarily empty')
        frame = self.frame()
        packet = MagicMock()
        packet.decode.return_value = [frame]
        capture.demux.side_effect = [empty, iter([packet])]
        stop = Event()
        frames = camera_frames(capture, stop)
        self.assertIs(next(frames), frame)
        stop.set()
        self.assertEqual(list(frames), [])
        self.assertEqual(capture.demux.call_count, 2)

    def test_stalled_camera_times_out_instead_of_reusing_frames_forever(self):
        capture = self.container()
        empty = MagicMock()
        empty.__next__.side_effect = av.error.BlockingIOError(errno.EAGAIN, 'temporarily empty')
        capture.demux.return_value = empty
        with patch('training.shared.devices.time.monotonic', side_effect=[0, 11]):
            with self.assertRaises(TimeoutError):
                next(camera_frames(capture, Event()))

    def test_end_of_camera_stream_is_reported_as_device_failure(self):
        capture = self.container()
        capture.demux.return_value = iter([])
        with self.assertRaises(ConnectionError):
            next(camera_frames(capture, Event()))

    def test_camera_worker_preserves_rgb_and_releases_the_device(self):
        camera = CameraCapture(0, 8, 6, 30)
        capture = self.container()
        frame = self.frame()

        def frames(container, stop):
            yield frame
            stop.set()

        with patch('training.shared.devices.open_camera') as opened:
            opened.return_value.__enter__.return_value = capture
            with patch('training.shared.devices.camera_frames', side_effect=frames):
                camera._capture()
            opened.return_value.__exit__.assert_called_once()
        result = camera.read()
        self.assertEqual(result.rgb.shape, (6, 8, 3))
        self.assertEqual(result.rgb.dtype, np.uint8)
        self.assertTrue((result.rgb[:, :, 0] == 255).all())
        self.assertTrue((result.rgb[:, :, 1:] == 0).all())
        self.assertTrue(camera.ready.is_set())
        self.assertEqual(camera.reported_fps, 30)

    def test_clockwise_rotation_changes_pixels_and_geometry(self):
        camera = CameraCapture(0, 8, 6, 30, rotation_clockwise=90)
        rgb = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
        with patch('training.shared.devices.open_camera') as opened:
            opened.return_value.__enter__.return_value = self.container()
            with patch('training.shared.devices.camera_frames', return_value=iter([frame])):
                camera._capture()
        np.testing.assert_array_equal(camera.read().rgb, np.rot90(rgb, k=-1))
        self.assertEqual(camera.read().rgb.shape, (8, 6, 3))
        self.assertTrue(camera.read().rgb.flags.c_contiguous)

    def test_wrong_capture_dimensions_are_rejected_before_recording(self):
        camera = CameraCapture(0, 8, 6, 30)
        with patch('training.shared.devices.open_camera') as opened:
            opened.return_value.__enter__.return_value = self.container()
            with patch('training.shared.devices.camera_frames', return_value=iter([self.frame(16, 12)])):
                camera._capture()
        with self.assertRaisesRegex(ConnectionError, 'different dimensions'):
            camera.read()
        self.assertTrue(camera.ready.is_set())

    def test_inspection_reports_actual_frame_dimensions(self):
        with patch('training.shared.devices.open_camera') as opened:
            opened.return_value.__enter__.return_value = self.container()
            with patch('training.shared.devices.camera_frames', return_value=iter([self.frame()])):
                profile = inspect_camera(0)
        self.assertEqual((profile['width'], profile['height']), (8, 6))
        self.assertEqual(profile['reported_frames_per_second'], 30)
        self.assertEqual(profile['backend'], 'pyav_avfoundation')

    def test_unknown_device_rate_is_safe_for_json_metadata(self):
        for rate in (None, float('nan'), float('inf'), Fraction(0, 1)):
            with self.subTest(rate=rate):
                capture = self.container()
                capture.streams.video[0].average_rate = rate
                with patch('training.shared.devices.open_camera') as opened:
                    opened.return_value.__enter__.return_value = capture
                    with patch('training.shared.devices.camera_frames', return_value=iter([self.frame()])):
                        profile = inspect_camera(0)
                self.assertIsNone(profile['reported_frames_per_second'])
                json.dumps(profile, allow_nan=False)

    def test_full_recorder_import_does_not_load_opencv_or_duplicate_classes(self):
        result = subprocess.run(
            [sys.executable, '-c',
             'import sys; from recording.server import create_app; assert "cv2" not in sys.modules'],
            cwd=Path(__file__).parents[1], capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('AVFFrameReceiver', result.stderr)
        self.assertNotIn('AVFAudioReceiver', result.stderr)


if __name__ == '__main__':
    unittest.main()
