"""Verify live inference ownership, preprocessing, snapshots and motion stops without hardware."""

from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import numpy as np
import torch

from models.policy import DapPolicy
from recording.server import create_app
from recording.session import RecorderSession
from shared.demo_devices import DemoFollower
from runtime.policy import PolicyDecision, PolicyRunner, checkpoint_catalog, checkpoint_path
from runtime.session import RuntimeSession
from shared.config import load_config
from shared.devices import CameraFrame, HoldingFollower, FollowerDevices
from shared.normalization import JOINT_NAMES
from shared.ownership import HardwareLease


PROJECT = Path(__file__).parents[1]


class FixedRunner:
    version = 'memory-policy-v1'

    def __init__(self):
        self.resets = 0
        self.target = [80.] * 5 + [100.]

    def reset(self):
        self.resets += 1

    def observe(self, *args):
        pass

    def predict(self, mode):
        return PolicyDecision(self.target, [0.] * 6, self.version)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = load_config(PROJECT/'configs/runtime.yaml')
        self.config['policy']['device'] = 'cpu'
        self.now = 10.
        self.devices = DemoFollower()
        original_observe = self.devices.observe

        def observe():
            frame, positions, _ = original_observe()
            frame.timestamp = self.now
            return frame, positions, self.now

        self.devices.observe = observe
        self.session = RuntimeSession(self.config,self.root/'pose.yaml',HardwareLease(self.root/'hardware.lock'),
                                      lambda settings:self.devices,lambda:self.now)
        self.owner = {'client_id':'test-owner'}
        self.runner = FixedRunner()
        self.session.handle_command('publish',{'runner':self.runner})
        self.session.handle_command('connect',{**self.owner,'settings':{}})
        self.addCleanup(self.session.disconnect)

    def capture_start(self):
        self.session.handle_command('capture_pose',self.owner)
        self.session.handle_command('start',{**self.owner,'mode':'mean'})

    def test_connect_holds_capture_does_not_move_start_requires_pose_and_current_position(self):
        self.assertEqual(self.session.snapshot()['phase'],'holding')
        self.assertTrue(self.session.snapshot()['hold_confirmed'])
        with self.assertRaisesRegex(ValueError,'Capture'):
            self.session.handle_command('start',{**self.owner,'mode':'mean'})
        before = self.devices.read_positions()
        self.session.handle_command('capture_pose',self.owner)
        self.assertEqual(self.devices.read_positions(),before)
        self.assertTrue((self.root/'pose.yaml').is_file())
        self.devices.positions[0] = 20.
        with self.assertRaisesRegex(ValueError,'Return'):
            self.session.handle_command('start',{**self.owner,'mode':'mean'})
        self.assertEqual(list(self.root.glob('*.mp4')),[])

    def test_clamps_requested_targets_and_stop_holds_measured_pose(self):
        self.capture_start()
        self.session.tick(self.now,1/30)
        self.assertEqual(self.devices.positions,[2.] * 5 + [52.])
        state = self.session.snapshot()
        self.assertEqual(state['requested_target'],self.runner.target)
        self.assertEqual(state['sent_target'],[2.] * 5 + [52.])
        self.assertEqual(state['diagnostics']['limited_commands'],1)
        self.devices.positions[0] = 1.5
        self.session.handle_command('stop',self.owner)
        self.assertEqual(self.devices.positions[0],1.5)
        self.assertEqual(self.session.snapshot()['phase'],'holding')
        self.assertGreaterEqual(self.runner.resets,3)

    def test_camera_staleness_heartbeat_loss_and_invalid_output_send_no_policy_command(self):
        self.capture_start()
        arm = CameraFrame(np.zeros((12,16,3),dtype=np.uint8),0,self.now-1)
        with patch.object(self.devices,'observe',return_value=(arm,self.devices.positions,self.now)), patch.object(self.devices,'secondary_frame',return_value=arm):
            with self.assertRaisesRegex(TimeoutError,'camera'):
                self.session.tick(self.now,1/30)
        self.assertEqual(self.devices.positions,[0.] * 5 + [50.])
        self.now += 3
        with self.assertRaisesRegex(TimeoutError,'heartbeat'):
            self.session.tick(self.now,1/30)
        self.session.heartbeat('test-owner')
        self.runner.target[0] = float('nan')
        with self.assertRaises(ValueError):
            self.session.tick(self.now,1/30)
        self.assertEqual(self.devices.positions,[0.] * 5 + [50.])

    def test_stop_cancel_during_prediction_and_camera_age_rechecked_before_send(self):
        self.capture_start()
        original = self.runner.predict
        def cancel(mode):
            self.session.motion_cancel.set()
            return original(mode)
        self.runner.predict = cancel
        self.session.tick(self.now,1/30)
        self.assertEqual(self.devices.positions[0],0.)
        self.session.motion_cancel.clear()
        def slow(mode):
            self.now += .5
            return original(mode)
        self.runner.predict = slow
        with self.assertRaisesRegex(TimeoutError,'stale'):
            self.session.tick(10.,1/30)
        self.assertEqual(self.devices.positions[0],0.)

    def test_return_to_pose_is_gradual_does_not_start_policy_and_calibration_is_bound(self):
        self.session.handle_command('capture_pose',self.owner)
        self.devices.positions[0] = 10.
        self.session.handle_command('reset',self.owner)
        self.session.tick(self.now,1/30)
        self.assertAlmostEqual(self.devices.positions[0],9.5)
        self.assertEqual(self.session.snapshot()['phase'],'resetting')
        self.session.handle_command('stop',self.owner)
        self.session.metadata['calibration']['follower']['source'] = 'different'
        with self.assertRaisesRegex(ValueError,'calibration'):
            self.session.handle_command('reset',self.owner)

    def test_return_interrupts_policy_and_finishes_holding_without_predictions(self):
        self.capture_start()
        self.devices.positions[0] = 10.
        with patch.object(self.runner, 'predict', wraps=self.runner.predict) as predict:
            self.session.handle_command('reset', self.owner)
            for _ in range(40):
                self.session.tick(self.now, 1/30)
                if self.session.snapshot()['phase'] == 'holding':
                    break
            predict.assert_not_called()
        self.assertEqual(self.session.snapshot()['phase'], 'holding')
        self.assertLessEqual(abs(self.devices.positions[0]), self.config['control']['reset_tolerance'])

    def test_return_advances_command_when_small_goals_do_not_move_motor(self):
        self.session.handle_command('capture_pose', self.owner)
        self.devices.positions[0] = 10.
        sent = []

        def sluggish_motor(targets):
            sent.append((self.devices.positions.copy(), targets.copy()))
            if abs(targets[0] - self.devices.positions[0]) >= 1.25:
                self.devices.positions = targets.copy()
            return targets.copy()

        with patch.object(self.devices, 'send_target', side_effect=sluggish_motor):
            self.session.handle_command('reset', self.owner)
            for _ in range(80):
                self.session.tick(self.now, 1/30)
                if self.session.snapshot()['phase'] == 'holding':
                    break
        self.assertEqual(self.session.snapshot()['phase'], 'holding')
        self.assertLessEqual(abs(self.devices.positions[0]), self.config['control']['reset_tolerance'])
        for measured, target in sent:
            self.assertLessEqual(max(abs(a-b) for a,b in zip(measured, target)), self.config['control']['max_target_delta'])
        goals = [target[0] for _, target in sent[:-1]]
        for previous, following in zip(goals, goals[1:]):
            self.assertLessEqual(abs(following-previous), self.config['control']['reset_speed_per_second']/30 + 1e-6)

    def test_active_snapshots_cannot_be_replaced_and_another_browser_cannot_control(self):
        self.capture_start()
        pending = FixedRunner()
        pending.version = 'next-policy-v2'
        self.session.handle_command('publish', {'runner': pending})
        self.assertIs(self.session.runner, self.runner)
        self.assertEqual(self.session.snapshot()['pending_policy_version'], pending.version)
        with self.assertRaises(ValueError):
            self.session.handle_command('load', {'runner': pending, **self.owner})
        with self.assertRaises(ValueError):
            self.session.handle_command('stop',{'client_id':'other-tab'})
        with self.assertRaises(ValueError):
            self.session.heartbeat('other-tab')

    def test_fault_holds_when_possible_and_reports_unconfirmed_hold_on_bus_failure(self):
        self.session.handle_fault(ConnectionError('camera failed'))
        self.assertEqual(self.session.snapshot()['phase'],'fault')
        self.assertTrue(self.session.snapshot()['hold_confirmed'])
        with patch.object(self.devices,'read_positions',side_effect=ConnectionError('bus lost')):
            self.session.handle_fault(ConnectionError('bus lost'))
        self.assertFalse(self.session.snapshot()['hold_confirmed'])
        self.assertIn('HOLD UNCONFIRMED',self.session.snapshot()['error'])

    def test_connect_recovers_stale_resources_after_failed_connection_cleanup(self):
        self.devices.disconnect()
        self.session.client_id = None
        self.session.update_state(connected=False, phase='fault', error='previous camera failure')
        self.session.handle_command('connect', {**self.owner, 'settings': {}})
        self.assertTrue(self.session.snapshot()['connected'])
        self.assertEqual(self.session.snapshot()['phase'], 'holding')
        self.assertIsNone(self.session.snapshot()['error'])

    def test_connect_reports_blocked_cleanup_without_opening_another_device(self):
        self.devices.disconnect()
        self.session.client_id = None
        self.session.update_state(connected=False, phase='fault')
        with patch.object(self.devices, 'disconnect', side_effect=TimeoutError('camera capture is blocked')), patch.object(self.session, 'factory') as factory:
            with self.assertRaisesRegex(RuntimeError, 'Previous device cleanup is incomplete.*camera capture is blocked'):
                self.session.handle_command('connect', {**self.owner, 'settings': {}})
            factory.assert_not_called()
        with patch.object(self.devices, 'send_target') as send:
            self.session.handle_fault(RuntimeError('camera cleanup blocked'))
            send.assert_not_called()
        self.assertNotIn('HOLD UNCONFIRMED', self.session.snapshot()['error'])

    def test_connect_still_rejects_a_real_connected_session(self):
        with self.assertRaisesRegex(ValueError, 'Disconnect the current session first'):
            self.session.handle_command('connect', {**self.owner, 'settings': {}})

    def test_cross_process_style_lease_excludes_another_session_and_survives_cleanup_failure(self):
        contender = HardwareLease(self.root/'hardware.lock')
        with self.assertRaises(ValueError):
            contender.acquire('BC recorder')
        with patch.object(self.devices,'disconnect',side_effect=ConnectionError('release failed')):
            with self.assertRaises(ConnectionError):
                self.session.disconnect()
        with self.assertRaises(ValueError):
            contender.acquire('BC recorder')
        self.session.disconnect()
        contender.acquire('BC recorder')
        contender.release('BC recorder')


class RuntimePolicyTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(PROJECT/'configs/bc.yaml')
        self.policy = DapPolicy(3)
        self.runner = PolicyRunner(self.policy,self.config,'memory-v1','cpu')

    def test_live_rotation_normalization_camera_order_and_startup_history(self):
        pixels = np.zeros((6,8,3),dtype=np.uint8)
        pixels[:,:,0] = 255
        mac = np.zeros((8,6,3),dtype=np.uint8)
        mac[:,:,1] = 255
        self.runner.observe(CameraFrame(pixels,0,1.),CameraFrame(mac,0,1.),[0.]*5+[50.],90)
        self.assertEqual(tuple(self.runner.images[0].shape),(2,3,168,224))
        torch.testing.assert_close(self.runner.images[0],self.runner.images[-1])
        self.assertEqual(self.runner.images[-1][0,0].mean().item(),1.)
        self.assertEqual(self.runner.images[-1][1,1].mean().item(),1.)
        torch.testing.assert_close(self.runner.positions[-1],torch.zeros(6))
        for mode in ('mean','sample'):
            action = self.runner.predict(mode)
            self.assertTrue(all(-1<=v<=1 for v in action.normalized))
            self.assertTrue(all(-100<=v<=100 for v in action.requested[:5]))
            self.assertTrue(0<=action.requested[-1]<=100)
        self.runner.reset()
        with self.assertRaises(ValueError):
            self.runner.predict('mean')

    def test_snapshot_does_not_share_weights_or_training_mode_with_learner(self):
        saved = next(self.runner.policy.parameters()).clone()
        with torch.no_grad():
            next(self.policy.parameters()).add_(100)
        torch.testing.assert_close(saved,next(self.runner.policy.parameters()))
        self.assertFalse(self.runner.policy.training)
        self.assertTrue(all(not p.requires_grad for p in self.runner.policy.parameters()))

    def test_checkpoint_catalog_scope_version_and_actor_only_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root/'run-a'
            run.mkdir()
            checkpoint = {'format_version':1,'config':self.config,'models':{'actor':self.policy.state_dict()},'progress':{'optimizer_step':42}}
            torch.save(checkpoint,run/'best_mae.pt')
            catalog = checkpoint_catalog(root)
            self.assertEqual(len(catalog['items']),1)
            self.assertIn('step 42',catalog['items'][0]['label'])
            with self.assertRaises(ValueError):
                checkpoint_path(root,'../outside.pt')
            loaded = PolicyRunner.from_checkpoint(root,'run-a/best_mae.pt','cpu')
            self.assertIn('step 42',loaded.version)
            torch.testing.assert_close(next(loaded.policy.parameters()),next(self.policy.parameters()))

    def test_holding_follower_sets_raw_goal_before_enabling_torque_and_uses_no_leader(self):
        robot = HoldingFollower.__new__(HoldingFollower)
        robot.bus = Mock()
        robot.bus.motors = {'gripper':Mock()}
        robot.bus.is_calibrated = True
        robot.config = Mock(position_p_coefficient=16,position_i_coefficient=0,position_d_coefficient=32,num_read_retries=2)
        robot.bus.sync_read.return_value = {'gripper':2048}
        robot.configure()
        calls = [call[0] for call in robot.bus.method_calls]
        self.assertLess(calls.index('sync_write'),calls.index('enable_torque'))
        robot.bus.sync_write.assert_called_once_with('Goal_Position',{'gripper':2048},normalize=False)
        self.assertFalse(FollowerDevices.requires_leader)


class RuntimeAPITests(unittest.TestCase):
    def test_demo_api_lease_pose_preview_and_real_policy_load(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            recorder = RecorderSession(root/'data')
            settings = load_config(PROJECT/'configs/runtime.yaml')
            settings['policy']['device'] = 'cpu'
            settings['control']['iterations_per_second'] = 10
            runtime = RuntimeSession(settings,root/'pose.yaml',recorder.lease,lambda config:DemoFollower())
            with TestClient(create_app(recorder,demo=True,profile_path=root/'connection.json',runtime=runtime)) as client:
                owner = {'client_id':'browser-a'}
                response = client.post('/api/inference/connect',json={**owner,'settings':{}})
                self.assertEqual(response.status_code,200,response.text)
                self.assertEqual(client.post('/api/connect',json={'width':320,'height':240}).status_code,409)
                with patch('recording.session.inspect_camera') as inspect:
                    self.assertEqual(client.post('/api/camera/inspect',json={'camera_index':0}).status_code,409)
                    inspect.assert_not_called()
                self.assertEqual(client.post('/api/inference/pose/capture',json=owner).status_code,200)
                self.assertEqual(client.post('/api/inference/preview/offer',json={
                    'client_id':'another-browser','sdp':'invalid','type':'offer'}).status_code,409)
                self.assertEqual(client.post('/api/inference/disconnect',json=owner).status_code,200)
                self.assertEqual(client.post('/api/inference/heartbeat',json=owner).status_code,409)
                self.assertFalse(list(root.rglob('*.mp4')))

    def test_repeated_deadlines_stop_worker_motion_and_require_explicit_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = load_config(PROJECT/'configs/runtime.yaml')
            settings['control']['browser_timeout_s'] = 10.
            devices = DemoFollower()
            runtime = RuntimeSession(settings,root/'pose.yaml',HardwareLease(root/'hardware.lock'),lambda config:devices)
            runner = FixedRunner()
            original = runner.predict

            def slow_prediction(mode):
                time.sleep(.045)
                return original(mode)

            runner.predict = slow_prediction
            runtime.start()
            try:
                owner = {'client_id':'browser-a'}
                runtime.publish(runner)
                runtime.request('connect',{**owner,'settings':{}})
                runtime.request('capture_pose',owner)
                runtime.request('start',{**owner,'mode':'mean'})
                deadline = time.monotonic()+2
                while runtime.snapshot()['phase'] != 'fault' and time.monotonic()<deadline:
                    time.sleep(.01)
                state = runtime.snapshot()
                self.assertEqual(state['phase'],'fault')
                self.assertIn('deadlines',state['error'])
                self.assertTrue(state['hold_confirmed'])
                stopped = devices.read_positions()
                time.sleep(.1)
                self.assertEqual(stopped,devices.read_positions())
            finally:
                runtime.close()


if __name__ == '__main__':
    unittest.main()
