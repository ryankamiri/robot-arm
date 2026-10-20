"""Verify binary policy delivery, immutable snapshots and between-attempt adoption."""

from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import torch

from models.policy import DapPolicy
from recording.server import create_app
from recording.session import RecorderSession
from runtime.policy import PolicyRunner
from runtime.session import RuntimeSession
from shared.config import load_config
from shared.demo_devices import DemoFollower
from shared.ownership import HardwareLease
from shared.policy_snapshots import MAX_SNAPSHOT_BYTES, decode_actor, encode_actor, send_actor
from training.sac.learner import SACLearner
from training.sac.run import LearnerRun


PROJECT = Path(__file__).resolve().parents[1]
PREPROCESSING = {'image_width': 224, 'image_height': 168}


class PolicyPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.actor = DapPolicy(3)
        self.settings = load_config(PROJECT / 'configs/runtime.yaml')
        self.settings['policy']['device'] = 'cpu'
        self.settings['control']['browser_timeout_s'] = 30.
        self.runtime = RuntimeSession(self.settings, self.root / 'pose.yaml', HardwareLease(self.root / 'hardware.lock'),
                                      lambda settings: DemoFollower())
        self.owner = {'client_id': 'browser-a'}

    def snapshot(self, step=0):
        return encode_actor(self.actor, PREPROCESSING, 'test-run', step)

    def runner(self, step=0):
        return PolicyRunner.from_snapshot(self.snapshot(step).content)

    def test_wire_snapshot_is_cpu_only_independent_and_deterministic(self):
        original = deepcopy(self.actor.state_dict())
        first = self.snapshot()
        self.assertEqual(first.digest, self.snapshot().digest)
        with torch.no_grad():
            next(self.actor.parameters()).add_(1)
        snapshot, loaded, _ = decode_actor(first.content)
        self.assertEqual(snapshot.version, first.version)
        for key, value in loaded.state_dict().items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)
            self.assertEqual(value.device.type, 'cpu')
        self.assertNotEqual(self.snapshot().digest, first.digest)

    def test_nonfinite_incompatible_and_oversized_snapshots_are_rejected(self):
        with self.assertRaises(ValueError):
            decode_actor(b'x' * (MAX_SNAPSHOT_BYTES + 1))
        for field, value in (('preprocessing', {'image_width': 1, 'image_height': 1}),
                             ('actor_updates', -1), ('run_id', '')):
            payload = torch.load(BytesIO(self.snapshot().content), weights_only=True)
            payload[field] = value
            buffer = BytesIO()
            torch.save(payload, buffer)
            with self.assertRaises(ValueError):
                decode_actor(buffer.getvalue())
        with torch.no_grad():
            next(self.actor.parameters()).fill_(float('nan'))
        with self.assertRaises(ValueError):
            decode_actor(self.snapshot().content)

    def test_running_policy_remains_unchanged_until_next_start(self):
        original = self.runner(0)
        self.runtime._command('publish', {'runner': original})
        self.runtime._command('connect', {**self.owner, 'settings': {}})
        self.addCleanup(self.runtime._disconnect)
        self.runtime._command('capture_pose', self.owner)
        self.runtime._command('start', {**self.owner, 'mode': 'sample'})
        first_pending = self.runner(1)
        newest = self.runner(2)
        self.runtime._command('publish', {'runner': first_pending})
        self.runtime._command('publish', {'runner': newest})
        self.assertIs(self.runtime.runner, original)
        self.assertIs(self.runtime.pending_runner, newest)
        self.assertEqual(self.runtime.snapshot()['phase'], 'running')
        self.runtime._command('stop', self.owner)
        self.assertIs(self.runtime.runner, original)
        self.runtime._command('start', {**self.owner, 'mode': 'sample'})
        self.assertIs(self.runtime.runner, newest)
        self.assertIsNone(self.runtime.pending_runner)
        self.assertEqual(self.runtime.snapshot()['policy_version'], newest.version)
        self.assertFalse(newest.images)

    @unittest.skipUnless(torch.backends.mps.is_available(), 'requires Apple MPS')
    def test_cpu_staged_snapshot_moves_to_mps_only_at_adoption(self):
        self.runtime.settings['policy']['device'] = 'mps'
        original = self.runner(0)
        self.runtime._command('publish', {'runner': original})
        self.runtime._command('connect', {**self.owner, 'settings': {}})
        self.addCleanup(self.runtime._disconnect)
        self.runtime._command('capture_pose', self.owner)
        self.runtime._command('start', {**self.owner, 'mode': 'mean'})
        pending = self.runner(1)
        self.runtime._command('publish', {'runner': pending})
        self.assertEqual(next(pending.policy.parameters()).device.type, 'cpu')
        self.assertEqual(next(original.policy.parameters()).device.type, 'mps')
        self.runtime._command('stop', self.owner)
        self.runtime._command('start', {**self.owner, 'mode': 'mean'})
        self.assertEqual(next(pending.policy.parameters()).device.type, 'mps')
        self.assertIs(self.runtime.runner, pending)

    def test_duplicate_deliveries_do_not_reset_and_older_or_conflicting_versions_fail(self):
        runner = self.runner(2)
        self.runtime._command('publish', {'runner': runner})
        with patch.object(runner, 'reset') as reset:
            self.runtime._command('publish', {'runner': self.runner(2)})
            reset.assert_not_called()
        self.assertIs(self.runtime.runner, runner)
        with self.assertRaises(ValueError):
            self.runtime._command('publish', {'runner': self.runner(1)})
        with torch.no_grad():
            next(self.actor.parameters()).add_(1)
        with self.assertRaises(ValueError):
            self.runtime._command('publish', {'runner': self.runner(2)})
        self.assertIs(self.runtime.runner, runner)

    def test_manual_load_clears_pending_and_duplicate_delivery_does_not_override_choice(self):
        self.runtime._update(phase='running')
        runner = self.runner(2)
        self.runtime._command('publish', {'runner': runner})
        self.runtime._update(phase='disconnected')
        chosen = PolicyRunner(DapPolicy(3), {'preprocessing': PREPROCESSING}, 'chosen-manually', 'cpu')
        self.runtime._command('load', {'runner': chosen})
        self.runtime._command('publish', {'runner': self.runner(2)})
        self.assertIs(self.runtime.runner, chosen)
        self.assertIsNone(self.runtime.pending_runner)

    def test_fastapi_binary_roundtrip_acknowledgement_and_rejection_leave_weights_intact(self):
        recorder = RecorderSession(self.root / 'episodes')
        with TestClient(create_app(recorder, demo=True, profile_path=self.root / 'connection.json', runtime=self.runtime)) as client:
            snapshot = self.snapshot(1)
            def deliver(request, timeout):
                response = client.post('/api/inference/policy', content=request.data, headers=dict(request.header_items()))
                self.assertEqual(response.status_code, 200, response.text)
                return BytesIO(response.content)
            with patch('shared.policy_snapshots.urlopen', side_effect=deliver):
                acknowledgement = send_actor(snapshot, 'http://127.0.0.1:8001/api/inference/policy', 10.)
            self.assertEqual(acknowledgement['publication_state'], 'active')
            self.assertEqual(acknowledgement['active_policy_version'], snapshot.version)
            self.assertEqual(client.post('/api/inference/policy', content=b'bad').status_code, 415)
            self.assertEqual(client.post('/api/inference/policy', content=b'bad', headers={'Content-Type': 'application/octet-stream'}).status_code, 409)
            self.assertEqual(client.post('/api/inference/policy', content=b'x' * (MAX_SNAPSHOT_BYTES + 1), headers={'Content-Type': 'application/octet-stream'}).status_code, 413)
            self.assertEqual(self.runtime.snapshot()['policy_version'], snapshot.version)
            self.assertFalse(list(self.root.rglob('*.pt')))
            self.assertIsNone(self.runtime.devices)

    def test_publication_acknowledges_pending_and_duplicate_while_running(self):
        recorder = RecorderSession(self.root / 'episodes')
        with TestClient(create_app(recorder, demo=True, profile_path=self.root / 'connection.json', runtime=self.runtime)) as client:
            self.runtime._update(phase='running')
            snapshot = self.snapshot(3)
            for _ in range(2):
                response = client.post('/api/inference/policy', content=snapshot.content, headers={'Content-Type': 'application/octet-stream'})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()['publication_state'], 'pending')
            self.assertIsNone(self.runtime.runner)
            self.assertEqual(self.runtime.snapshot()['pending_policy_version'], snapshot.version)

    def test_learner_publication_has_no_file_io_and_failure_preserves_recovery_state(self):
        config = load_config(PROJECT / 'configs/sac.yaml')
        learner = SACLearner(self.actor, config, 'test-reward')
        run = LearnerRun(learner, self.root / 'run')
        def received(snapshot, url, timeout):
            return {'received_policy_version': snapshot.version, 'sha256': snapshot.digest, 'publication_state': 'pending'}
        with patch('training.sac.run.send_actor', side_effect=received):
            self.assertEqual(run.publish()['publication_state'], 'pending')
        self.assertFalse(list(run.output.rglob('*')))
        with patch('training.sac.run.send_actor', side_effect=RuntimeError('backend down')):
            with self.assertRaisesRegex(RuntimeError, 'backend down'):
                run.publish()
        self.assertTrue((run.output / 'learner' / 'latest.pt').exists())

    def test_client_rejects_wrong_acknowledgement_and_remote_destination(self):
        with patch('shared.policy_snapshots.urlopen', return_value=BytesIO(json.dumps({'received_policy_version': 'wrong'}).encode())):
            with self.assertRaises(ValueError):
                send_actor(self.snapshot(), 'http://localhost:8001/api/inference/policy', 1.)
        with self.assertRaises(ValueError):
            send_actor(self.snapshot(), 'http://example.com/api/inference/policy', 1.)


if __name__ == '__main__':
    unittest.main()
