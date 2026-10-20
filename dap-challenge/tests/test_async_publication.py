"""Prove HTTP delivery cannot stall optimization and queued snapshots stay bounded."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
import unittest
from unittest.mock import patch

import torch

from models.policy import DapPolicy
from shared.config import load_config
from shared.policy_snapshots import ActorSnapshot
from training.sac.learner import SACLearner
from training.sac.replay import ScoredEpisode
from training.sac.run import LearnerRun
from training.shared.publication import AsyncPolicyPublisher


PROJECT = Path(__file__).resolve().parents[1]


def snapshot(step: int) -> ActorSnapshot:
    return ActorSnapshot(str(step).encode(), 'synthetic', step, str(step))


def acknowledgement(value: ActorSnapshot) -> dict:
    return {'received_policy_version': value.version, 'sha256': value.digest, 'publication_state': 'pending'}


class AsyncPublicationTests(unittest.TestCase):
    def test_slow_send_retains_only_newest_pending_version_and_flushes_on_close(self):
        entered, release = Event(), Event()
        received = []
        def send(value, url, timeout):
            received.append(value.actor_updates)
            if len(received) == 1:
                entered.set()
                if not release.wait(5):
                    raise TimeoutError('test sender was never released')
            return acknowledgement(value)
        publisher = AsyncPolicyPublisher('http://localhost/api', 1.)
        with patch('training.shared.publication.send_actor', side_effect=send):
            try:
                publisher.submit(snapshot(1))
                self.assertTrue(entered.wait(2))
                publisher.submit(snapshot(2))
                publisher.submit(snapshot(3))
                publisher.submit(snapshot(3))
                state = publisher.snapshot()
                self.assertEqual(state['in_flight'], snapshot(1).version)
                self.assertEqual(state['pending'], snapshot(3).version)
                self.assertEqual(received, [1])
            finally:
                release.set()
                publisher.close()
        self.assertEqual(received, [1, 3])
        self.assertEqual(publisher.snapshot()['acknowledgement'], acknowledgement(snapshot(3)))
        self.assertIsNone(publisher.snapshot()['pending'])
        self.assertFalse(publisher.worker.is_alive())
        with self.assertRaises(RuntimeError):
            publisher.submit(snapshot(4))

    def test_learner_updates_during_blocked_delivery_and_wire_weights_remain_frozen(self):
        config = load_config(PROJECT / 'configs/sac.yaml')
        config['training']['batch_size'] = 2
        config['training']['warmup'].update(minimum_scored_episodes=1, minimum_transitions=1, critic_only_updates=1)
        config['preprocessing'] = {'image_height': 9, 'image_width': 11}
        learner = SACLearner(DapPolicy(3), config, 'test-reward')
        entered, release = Event(), Event()
        delivered = []
        def send(value, url, timeout):
            delivered.append(value)
            entered.set()
            if not release.wait(5):
                raise TimeoutError('test sender was never released')
            return acknowledgement(value)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            episode = ScoredEpisode('synthetic', 'base-policy', 'test-reward', 1.,
                                    torch.zeros(4, 2, 3, 9, 11, dtype=torch.uint8), torch.zeros(4, 6),
                                    torch.zeros(3, 6), torch.zeros(3, 6))
            learner.ingest(episode.save(root))
            for _ in range(3):
                learner.update()
            run = LearnerRun(learner, root / 'run')
            with patch('training.shared.publication.send_actor', side_effect=send):
                try:
                    run.publish()
                    self.assertTrue(entered.wait(2))
                    before = learner.progress['critic_updates']
                    content = delivered[0].content
                    old_weights = torch.load(BytesIO(content), weights_only=True)['weights']
                    learner.update()
                    learner.update()
                    self.assertEqual(learner.progress['critic_updates'], before + 2)
                    self.assertEqual(learner.progress['actor_updates'], 2)
                    self.assertFalse(release.is_set())
                    self.assertTrue(any(not torch.equal(value, learner.actor.state_dict()[key]) for key, value in old_weights.items()))
                    self.assertEqual(delivered[0].content, content)
                    queued = run.publish()
                    self.assertEqual(run.publisher.snapshot()['pending'], queued['policy_version'])
                    self.assertFalse(list(run.output.rglob('*')))
                finally:
                    release.set()
                    run.close()
            self.assertEqual([value.actor_updates for value in delivered], [1, 2])

    def test_worker_failure_is_observable_and_preserves_learner_on_main_thread(self):
        config = load_config(PROJECT / 'configs/sac.yaml')
        with TemporaryDirectory() as directory:
            run = LearnerRun(SACLearner(DapPolicy(3), config, 'test-reward'), Path(directory))
            failed = Event()
            def fail(*args):
                failed.set()
                raise ConnectionError('receiver unavailable')
            with patch('training.shared.publication.send_actor', side_effect=fail):
                run.publish()
                self.assertTrue(failed.wait(2))
                with self.assertRaisesRegex(RuntimeError, 'receiver unavailable'):
                    run.close()
            with self.assertRaisesRegex(RuntimeError, 'receiver unavailable'):
                run.check_publication()
            self.assertTrue((Path(directory) / 'learner' / 'latest.pt').exists())
            self.assertIn('receiver unavailable', run.publisher.snapshot()['error'])
            self.assertFalse(run.publisher.worker.is_alive())

    def test_enqueue_failure_after_poll_still_saves_the_current_learner(self):
        config = load_config(PROJECT / 'configs/sac.yaml')
        with TemporaryDirectory() as directory:
            run = LearnerRun(SACLearner(DapPolicy(3), config, 'test-reward'), Path(directory))
            with patch.object(run.publisher, 'submit', side_effect=RuntimeError('publication failed during cloning')):
                with self.assertRaisesRegex(RuntimeError, 'during cloning'):
                    run.publish()
            self.assertTrue((Path(directory) / 'learner' / 'latest.pt').exists())
            run.close()

    def test_abort_discards_pending_send_and_joins_the_in_flight_worker(self):
        entered, release = Event(), Event()
        received = []
        def send(value, url, timeout):
            received.append(value.actor_updates)
            entered.set()
            if not release.wait(5):
                raise TimeoutError('test sender was never released')
            return acknowledgement(value)
        publisher = AsyncPolicyPublisher('http://localhost/api', 1.)
        with patch('training.shared.publication.send_actor', side_effect=send):
            publisher.submit(snapshot(1))
            self.assertTrue(entered.wait(2))
            publisher.submit(snapshot(2))
            closer = Thread(target=lambda: publisher.close(flush=False))
            closer.start()
            with publisher.condition:
                self.assertTrue(publisher.condition.wait_for(lambda: publisher.closing, timeout=2))
            release.set()
            closer.join(timeout=2)
        self.assertFalse(closer.is_alive())
        self.assertFalse(publisher.worker.is_alive())
        self.assertEqual(received, [1])


if __name__ == '__main__':
    unittest.main()
