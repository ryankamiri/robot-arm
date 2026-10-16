"""Exercise real recorder files and API state changes without moving hardware."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import av
from fastapi.testclient import TestClient
import pyarrow.parquet as pq
import torch

from recording.server import create_app
from recording.session import RecorderSession
from recording.storage import EpisodeStore, EpisodeWriter
from training.shared.devices import DemoDevices, SO101Devices
from training.shared.episodes import discover_episodes, load_episode
from training.shared.normalization import JOINT_NAMES, normalize_controller, denormalize_controller


class ControlledDevices(DemoDevices):
    def connect(self):
        self.commands = 0
        self.closed = False
        self.fail = False
        self.delay = 0.0
        return super().connect()

    def observe(self):
        if self.fail:
            raise ConnectionError('test camera disconnected')

        time.sleep(self.delay)
        return super().observe()

    def command(self):
        self.commands += 1
        return [30.0, 0.0, 0.0, 0.0, 0.0, 50.0]

    def disconnect(self):
        self.closed = True


class DualCameraDevices(ControlledDevices):
    def connect(self):
        metadata = super().connect()
        metadata['camera']['rotation_clockwise'] = 0
        metadata['mac_camera'] = {'rotation_clockwise': 0, 'width': 320, 'height': 240, 'timestamp_basis': 'synthetic_host_monotonic'}
        return metadata

    def observe(self):
        result = super().observe()
        self.mac_frame = result[0]
        return result

    def secondary_frame(self):
        return self.mac_frame


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'data'
        self.device = ControlledDevices()
        self.session = RecorderSession(self.root, lambda config: self.device)
        self.client = TestClient(create_app(self.session, demo=True))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def post(self, route, payload=None):
        response = self.client.post('/api/' + route, json=payload or {})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def connect(self):
        self.post('connect', {'width': 320, 'height': 240})

    def wait_for(self, predicate):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            state = self.client.get('/api/status').json()
            if predicate(state):
                return state
            time.sleep(0.01)
        self.fail('recorder did not reach expected state')

    def record(self):
        state = self.post('recording/start')
        episode_id = state['active_episode_id']
        self.wait_for(lambda state: state['recorded_rows'] >= 4)
        self.post('recording/stop')
        return episode_id

    def test_two_camera_recording_saves_previews_and_keeps_both_videos(self):
        self.session.factory = lambda config: DualCameraDevices()
        self.connect()
        episode_id = self.record()
        pending = self.session.store.path(episode_id)
        rows = pq.read_table(pending / 'steps.parquet').to_pydict()
        self.assertEqual(len(rows['mac_frame_index']), len(rows['frame_index']))
        self.assertTrue(all(m <= t for m, t in zip(rows['mac_camera_timestamp_s'], rows['timestamp_s'])))
        for query, filename in [('', 'video.mp4'), ('?camera=mac', 'mac_video.mp4')]:
            response = self.client.get(f'/api/episodes/{episode_id}/video{query}')
            self.assertEqual(response.status_code, 200)
            with av.open(str(pending / filename)) as video:
                frames = list(video.decode(video=0))
                self.assertGreater(len(frames), 0)
        self.post('episodes/keep', {'episode_id': episode_id, 'name': 'Two views'})
        accepted = self.root / episode_id
        self.assertTrue((accepted / 'mac_video.mp4').exists())
        metadata = json.loads((accepted / 'metadata.json').read_text())
        self.assertEqual(metadata['mac_camera']['video_file'], 'mac_video.mp4')

    def test_two_camera_take_cannot_be_kept_if_mac_video_is_missing(self):
        self.session.factory = lambda config: DualCameraDevices()
        self.connect()
        episode_id = self.record()
        (self.session.store.path(episode_id) / 'mac_video.mp4').unlink()
        response = self.client.post('/api/episodes/keep', json={'episode_id': episode_id})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.session.store.metadata(episode_id)['status'], 'review')

    def test_keep_preserves_optional_name_and_notes_without_renaming_directory(self):
        self.connect()
        episode_id = self.record()
        self.post('episodes/keep', {'episode_id': episode_id, 'name': '  First clean dap  ',
                                  'notes': '  Slow approach; good contact.  '})
        accepted = self.root / episode_id
        metadata = json.loads((accepted / 'metadata.json').read_text())
        self.assertEqual(metadata['name'], 'First clean dap')
        self.assertEqual(metadata['notes'], 'Slow approach; good contact.')
        self.assertEqual(metadata['episode_id'], episode_id)
        self.assertEqual(metadata['status'], 'accepted')
        self.assertTrue((accepted / 'video.mp4').exists())
        self.assertEqual(discover_episodes(self.root), [accepted])

    def test_keep_roundtrip_stores_sent_controller_targets_and_normalizes_on_load(self):
        self.device = DualCameraDevices()
        self.connect()
        episode_id = self.record()
        pending = self.session.store.path(episode_id)
        table = pq.read_table(pending / 'steps.parquet').to_pydict()
        self.assertTrue(all(row == [30, 0, 0, 0, 0, 50] for row in table['actions']))
        self.assertTrue(all(row == [0, 0, 0, 0, 0, 50] for row in table['positions']))
        self.assertTrue(all(action >= obs >= camera for action, obs, camera in zip(
            table['action_timestamp_s'], table['timestamp_s'], table['camera_timestamp_s'],
        )))
        with self.assertRaises(ValueError):
            discover_episodes(self.root)
        self.assertEqual(self.client.get(f'/api/episodes/{episode_id}/video').status_code, 200)
        before = self.device.commands
        self.wait_for(lambda state: self.device.commands > before + 1)
        self.assertEqual(self.session.store.metadata(episode_id)['row_count'], len(table['actions']))
        self.post('episodes/keep', {'episode_id': episode_id})
        self.assertEqual(self.client.get('/api/status').json()['phase'], 'reset')
        episode = load_episode(self.root / episode_id, 8, 6)
        torch.testing.assert_close(episode.actions[:, 0], torch.full((len(episode),), 0.3))
        torch.testing.assert_close(episode.actions[:, -1], torch.zeros(len(episode)))
        self.assertEqual(discover_episodes(self.root), [self.root / episode_id])
        self.post('reset/ready')
        self.assertEqual(self.client.get('/api/status').json()['phase'], 'ready')

    def test_video_finalization_does_not_pause_leader_following(self):
        self.connect()
        self.post('recording/start')
        self.wait_for(lambda state: state['recorded_rows'] >= 2)
        finish = self.session.writer.finish

        def slow_finish(error=None):
            time.sleep(0.25)
            finish(error)

        with patch.object(self.session.writer, 'finish', side_effect=slow_finish):
            with ThreadPoolExecutor(max_workers=1) as requests:
                response = requests.submit(self.client.post, '/api/recording/stop')
                self.wait_for(lambda state: state['phase'] == 'finalizing')
                before = self.device.commands
                self.wait_for(lambda state: self.device.commands >= before + 3)
                self.assertFalse(response.done())
                self.assertEqual(response.result(timeout=3).status_code, 200)
        self.assertEqual(self.client.get('/api/status').json()['phase'], 'review')

    def test_external_web_page_cannot_disconnect_local_devices(self):
        self.connect()
        response = self.client.post('/api/disconnect', headers={'Origin': 'https://example.com'})
        self.assertEqual(response.status_code, 403)
        self.assertFalse(self.device.closed)
        self.assertTrue(self.client.get('/api/status').json()['connected'])

    def test_state_guards_discard_and_disconnect_release(self):
        self.assertEqual(self.client.post('/api/recording/start').status_code, 409)
        self.connect()
        episode_id = self.post('recording/start')['active_episode_id']
        self.assertEqual(self.client.post('/api/episodes/discard', json={'episode_id': episode_id}).status_code, 409)
        self.wait_for(lambda state: state['recorded_rows'] >= 2)
        self.post('disconnect')
        self.assertTrue(self.device.closed)
        self.wait_for(lambda state: self.session.store.metadata(episode_id)['status'] == 'incomplete')
        self.assertEqual(self.session.store.metadata(episode_id)['status'], 'incomplete')
        self.assertEqual(self.client.post('/api/episodes/keep', json={'episode_id': episode_id}).status_code, 409)
        self.post('episodes/discard', {'episode_id': episode_id})
        self.assertFalse((self.session.store.pending / episode_id).exists())
        self.assertEqual(self.client.post('/api/episodes/discard', json={'episode_id': '..'}).status_code, 409)

    def test_fault_stops_teleoperation_and_preserves_incomplete_take(self):
        self.connect()
        episode_id = self.post('recording/start')['active_episode_id']
        self.wait_for(lambda state: state['recorded_rows'] >= 2)
        self.device.fail = True
        state = self.wait_for(lambda state: state['phase'] == 'error')
        self.assertFalse(state['connected'])
        self.assertIn('camera disconnected', state['error'])
        self.assertTrue(self.device.closed)
        self.wait_for(lambda state: self.session.store.metadata(episode_id)['status'] == 'incomplete')
        self.assertEqual(self.session.store.metadata(episode_id)['status'], 'incomplete')
        self.assertTrue((self.session.store.path(episode_id) / 'steps.parquet').exists())

    def test_slow_ticks_are_measured_without_stopping_the_recording(self):
        self.connect()
        self.device.delay = 0.045
        episode_id = self.record()
        metadata = self.session.store.metadata(episode_id)
        self.assertEqual(metadata['status'], 'review')
        self.assertGreater(metadata['diagnostics']['missed_deadlines'], 0)
        self.assertGreater(metadata['diagnostics']['max_cycle_s'], 1 / 30)
        self.assertGreater((self.session.store.path(episode_id) / 'timing.jsonl').stat().st_size, 0)

    def test_recovery_keeps_unfinished_rows_and_never_publishes(self):
        store = EpisodeStore(Path(self.temporary.name) / 'recovery')
        writer = EpisodeWriter(store, self.device.connect(), 30)
        for _ in range(3):
            frame, positions, observed = self.device.observe()
            writer.append(frame.rgb, frame.sequence, frame.timestamp, observed, time.monotonic(),
                          positions, self.device.command())
            time.sleep(0.001)
        writer.container.close()
        writer.journal.close()
        writer.timing.close()
        recovered = EpisodeStore(store.root)
        metadata = recovered.metadata(writer.episode_id)
        self.assertEqual(metadata['status'], 'incomplete')
        self.assertEqual(pq.read_table(recovered.path(writer.episode_id) / 'steps.parquet').num_rows, 3)
        with self.assertRaises(ValueError):
            recovered.keep(writer.episode_id)
        with self.assertRaises(ValueError):
            discover_episodes(store.root)

    def test_video_preserves_nonuniform_frame_times(self):
        store = EpisodeStore(Path(self.temporary.name) / 'variable')
        writer = EpisodeWriter(store, self.device.connect(), 30)
        origin = writer.started + 0.001
        for index, offset in enumerate((0, 0.04, 0.13)):
            frame, positions, _ = self.device.observe()
            camera_time = origin + offset
            writer.append(frame.rgb, index, camera_time, camera_time + 0.001,
                          camera_time + 0.002, positions, self.device.command())
        writer.finish()
        with av.open(str(writer.path / 'video.mp4')) as video:
            times = [float(frame.pts * frame.time_base) for frame in video.decode(video=0)]
        self.assertAlmostEqual(times[1] - times[0], 0.04, places=3)
        self.assertAlmostEqual(times[2] - times[0], 0.13, places=3)

    def test_repeated_camera_frame_uses_one_encoded_frame_and_one_capture_time(self):
        store = EpisodeStore(Path(self.temporary.name) / 'repeats')
        writer = EpisodeWriter(store, self.device.connect(), 30)
        frame, positions, observed = self.device.observe()
        for offset in (0.001, 0.002):
            writer.append(frame.rgb, frame.sequence, frame.timestamp, observed + offset,
                          observed + offset + 0.001, positions, self.device.command())
        writer.finish()
        rows = pq.read_table(writer.path / 'steps.parquet').to_pydict()
        self.assertEqual(rows['frame_index'], [0, 0])
        self.assertEqual(rows['camera_timestamp_s'][0], rows['camera_timestamp_s'][1])
        self.assertEqual(writer.metadata['diagnostics']['repeated_frames'], 1)
        with av.open(str(writer.path / 'video.mp4')) as video:
            self.assertEqual(len(list(video.decode(video=0))), 1)

    def test_second_backend_cannot_recover_or_take_ownership_of_active_recording(self):
        self.connect()
        episode_id = self.post('recording/start')['active_episode_id']
        self.wait_for(lambda state: state['recorded_rows'] >= 2)
        second = RecorderSession(self.root)
        with self.assertRaises(RuntimeError):
            second.start()
        self.assertEqual(self.session.store.metadata(episode_id)['status'], 'recording')
        self.post('recording/stop')

    def test_adapter_uses_actual_sent_action_not_requested_action(self):
        adapter = SO101Devices({})
        requested = {f'{name}.pos': 70 for name in JOINT_NAMES}
        sent = {f'{name}.pos': 30 for name in JOINT_NAMES}
        adapter.leader = Mock()
        adapter.follower = Mock()
        adapter.leader.get_action.return_value = requested
        adapter.follower.send_action.return_value = sent
        self.assertEqual(adapter.command(), [30.0] * 6)
        adapter.follower.send_action.assert_called_once_with(requested)

    def test_normalization_is_invertible_and_does_not_clip(self):
        values = torch.tensor([[-100, 0, 100, 20, -20, 50]], dtype=torch.float32)
        normalized = normalize_controller(values, JOINT_NAMES)
        torch.testing.assert_close(denormalize_controller(normalized, JOINT_NAMES), values)
        self.assertEqual(normalized[0, -1].item(), 0)
        with self.assertRaises(ValueError):
            normalize_controller(values + 200, JOINT_NAMES)


if __name__ == '__main__':
    unittest.main()
