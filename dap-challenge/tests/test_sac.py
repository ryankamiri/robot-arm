"""Verify SAC math, transition histories, finite budgets, exact resume and MPS updates."""

import argparse
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import torch

from models.critic import TwinQValues
from models.policy import DapPolicy
from shared.config import load_config, save_config
from runtime.policy import PolicyRunner, checkpoint_catalog
from training.sac.config import validate_config
from training.sac.learner import SACLearner
from training.sac.loss import actor_loss, critic_loss, critic_target, temperature_loss
from training.sac.replay import EpisodeReplay, ScoredEpisode
from training.sac.run import LearnerRun, initialize_learner
from training.sac.train import train
from training.shared.utils import training_directory


CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'sac.yaml'


def small_config():
    config = load_config(CONFIG)
    config['model']['history_length'] = 3
    config['preprocessing'] = dict(image_width=11, image_height=9)
    config['training']['batch_size'] = 2
    config['training']['publication']['enabled'] = False
    config['training']['warmup'].update(minimum_scored_episodes=1, minimum_transitions=1, critic_only_updates=2)
    config['training']['checkpoint'].update(every_critic_updates=2, keep_periodic=2)
    config['training']['publication']['every_actor_updates'] = 1
    return config


def episode(name='take', count=3):
    images = torch.arange(count + 1, dtype=torch.uint8).reshape(-1, 1, 1, 1, 1).expand(-1, 2, 3, 9, 11).clone()
    positions = torch.arange(count + 1).float().reshape(-1, 1).expand(-1, 6).clone() / 10
    return ScoredEpisode(name, 'bc-source-hash', 'rubric-v1', 1., images, positions,
                         torch.full((count, 6), .3), torch.full((count, 6), .1))


class SACMathTests(unittest.TestCase):
    def test_terminal_reward_and_detached_soft_bootstrap(self):
        next_q = torch.tensor([10., 10.], requires_grad=True)
        targets = critic_target(torch.tensor([1., 2.]), torch.tensor([True, False]),
                                TwinQValues(next_q, next_q + 1), torch.tensor([-2., -2.]), torch.tensor(.5), .9)
        torch.testing.assert_close(targets, torch.tensor([1., 11.9]))
        self.assertFalse(targets.requires_grad)
        self.assertEqual(critic_loss(TwinQValues(torch.tensor([2., 4.]), torch.tensor([3., 5.])), torch.tensor([1., 2.])).item(), 9.)

    def test_actor_gradient_passes_through_q_action(self):
        action = torch.tensor([.2, .4], requires_grad=True)
        loss = actor_loss(TwinQValues(2 * action, 3 * action), torch.zeros(2), torch.tensor(.1))
        loss.backward()
        torch.testing.assert_close(action.grad, torch.full((2,), -1.))

    def test_temperature_direction(self):
        log_alpha = torch.tensor(-4., requires_grad=True)
        temperature_loss(log_alpha, torch.tensor([10.]), -6.).backward()
        self.assertLess(log_alpha.grad.item(), 0)
        log_alpha.grad = None
        temperature_loss(log_alpha, torch.tensor([0.]), -6.).backward()
        self.assertGreater(log_alpha.grad.item(), 0)


class SACReplayTests(unittest.TestCase):
    def test_sample_uses_requested_actions_terminal_reward_and_causal_padding(self):
        with TemporaryDirectory() as directory:
            replay = EpisodeReplay(3, 'rubric-v1', 1)
            path = episode().save(Path(directory))
            self.assertEqual(replay.add(path), 3)
            self.assertEqual(replay.add(path), 0)
            batch = replay.sample(100, torch.device('cpu'))
            torch.testing.assert_close(batch.actions, torch.full((100, 6), .3))
            current = (batch.images[:, -1, 0, 0, 0, 0] * 255).round().long()
            for row, time in enumerate(current.tolist()):
                expected = torch.arange(time - 2, time + 1).clamp_min(0).float() / 255
                torch.testing.assert_close(batch.images[row, :, 0, 0, 0, 0], expected)
                expected_next = torch.arange(time - 1, time + 2).clamp_min(0).float() / 255
                torch.testing.assert_close(batch.next_images[row, :, 0, 0, 0, 0], expected_next)
                self.assertEqual(batch.terminated[row].item(), time == 2)
                self.assertEqual(batch.rewards[row].item(), float(time == 2))

    def test_episode_boundaries_never_mix_and_cache_stays_bounded(self):
        with TemporaryDirectory() as directory:
            replay = EpisodeReplay(3, 'rubric-v1', 1)
            for name in ('a', 'b'):
                item = episode(name, 1)
                if name == 'b':
                    item.images += 50
                replay.add(item.save(Path(directory)))
            batch = replay.sample(100, torch.device('cpu'))
            self.assertTrue(torch.equal(batch.images[:, 0], batch.images[:, 2]))
            self.assertLessEqual(len(replay.cache), 1)
            self.assertTrue(batch.terminated.all())

    def test_persistence_rejects_bad_data_and_unscored_attempts(self):
        for field, value in (('outcome', 'aborted'), ('reward', float('nan')), ('reward_version', ''),
                             ('requested_actions', torch.full((3, 6), 2.)), ('positions', torch.zeros(3, 6))):
            with self.subTest(field=field):
                item = episode()
                setattr(item, field, value)
                with self.assertRaises(ValueError):
                    item.validate()
        with TemporaryDirectory() as directory:
            path = episode().save(Path(directory))
            with self.assertRaises(FileExistsError):
                episode().save(Path(directory))
            with self.assertRaises(ValueError):
                EpisodeReplay(3, 'different-reward', 1).add(path)

    def test_rejects_duplicate_identity_and_changed_resume_source(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = episode().save(root)
            replay = EpisodeReplay(3, 'rubric-v1', 1)
            replay.add(path)
            item = episode()
            item.reward = .5
            changed = item.save(root / 'other')
            with self.assertRaises(ValueError):
                replay.add(changed)
            with self.assertRaises(ValueError):
                EpisodeReplay(3, 'rubric-v1', 1).add(changed, replay.records[0]['sha256'])


class SACLearnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        torch.manual_seed(2)
        self.config = small_config()
        self.learner = SACLearner(DapPolicy(3), self.config, 'rubric-v1')
        self.path = episode().save(self.root)
        self.learner.ingest(self.path)

    def tearDown(self):
        self.directory.cleanup()

    def test_warmup_then_two_critic_rounds_per_actor_and_budget_exhaustion(self):
        before = deepcopy(self.learner.actor.state_dict())
        for _ in range(2):
            self.learner.update()
        for key, value in before.items():
            torch.testing.assert_close(value, self.learner.actor.state_dict()[key], rtol=0, atol=0)
        self.learner.update()
        self.assertEqual(self.learner.progress['actor_updates'], 0)
        self.learner.update()
        self.assertEqual(self.learner.progress['actor_updates'], 1)
        self.assertTrue(any(not torch.equal(value, self.learner.actor.state_dict()[key]) for key, value in before.items()))
        while self.learner.ready:
            self.learner.update()
        self.assertEqual(self.learner.progress['critic_updates'], 6)
        self.assertEqual(self.learner.progress['actor_updates'], 2)
        self.assertFalse(self.learner.ready)
        self.assertEqual(self.learner.ingest(self.path), 0)
        self.assertEqual(self.learner.progress['earned_updates'], 6)
        with self.assertRaises(RuntimeError):
            self.learner.update()

    def test_both_startup_gates_are_required(self):
        self.learner.settings['warmup'].update(minimum_scored_episodes=2, minimum_transitions=5)
        self.assertFalse(self.learner.ready)
        self.learner.ingest(episode('second', 1).save(self.root))
        self.assertFalse(self.learner.ready)
        self.learner.ingest(episode('third', 1).save(self.root))
        self.assertTrue(self.learner.ready)

    def test_target_update_matches_polyak_and_targets_receive_no_gradients(self):
        before = deepcopy(self.learner.targets.state_dict())
        self.learner.update()
        tau = self.config['training']['target_critics']['tau']
        for key, value in self.learner.targets.state_dict().items():
            torch.testing.assert_close(value, torch.lerp(before[key], self.learner.critics.state_dict()[key], tau))
        self.assertTrue(all(parameter.grad is None for parameter in self.learner.targets.parameters()))

    def test_actor_update_does_not_modify_critics_or_leave_critic_grads(self):
        before = deepcopy(self.learner.critics.state_dict())
        batch = self.learner.replay.sample(2, torch.device('cpu'))
        self.learner._update_actor(batch)
        for key, value in self.learner.critics.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in self.learner.critics.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in self.learner.critics.parameters()))

    def test_exact_resume_restores_rng_optimizers_targets_and_budget(self):
        for _ in range(4):
            self.learner.update()
        path = self.learner.save(self.root / 'resume.pt')
        expected_metrics = self.learner.update()
        expected = deepcopy(self.learner.actor.state_dict())
        resumed = SACLearner(DapPolicy(3), self.config, 'rubric-v1')
        resumed.resume(path)
        actual_metrics = resumed.update()
        self.assertEqual(expected_metrics, actual_metrics)
        self.assertEqual(resumed.progress, self.learner.progress)
        for key, value in resumed.actor.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        self.assertEqual(resumed.ingest(self.path), 0)

    def test_critic_import_skips_warmup_but_requires_replay_and_matching_reward(self):
        self.learner.update()
        path = self.learner.save(self.root / 'critic.pt')
        trained_config = deepcopy(self.config)
        trained_config['initialization']['critic_is_base'] = False
        trained_config['training']['warmup']['enabled'] = False
        imported = SACLearner(DapPolicy(3), trained_config, 'rubric-v1')
        imported.import_critics(path)
        self.assertFalse(imported.ready)
        imported.ingest(self.path)
        imported.update()
        imported.update()
        self.assertEqual(imported.progress['warmup_updates'], 0)
        self.assertEqual(imported.progress['actor_updates'], 1)
        with self.assertRaises(ValueError):
            SACLearner(DapPolicy(3), self.config, 'other').import_critics(path)

    def test_nonfinite_stops_without_overwriting_good_checkpoint(self):
        path = self.learner.save(self.root / 'good.pt')
        original = path.read_bytes()
        batch = self.learner.replay.sample(2, torch.device('cpu'))
        batch.rewards[:] = float('nan')
        with self.assertRaises(FloatingPointError):
            self.learner.update(batch)
        self.assertFalse(self.learner.ready)
        with self.assertRaises(RuntimeError):
            self.learner.save(path)
        self.assertEqual(path.read_bytes(), original)

    def test_resume_rejects_config_drift_and_modified_replay(self):
        path = self.learner.save(self.root / 'resume.pt')
        changed = deepcopy(self.config)
        changed['training']['discount'] = .9
        with self.assertRaises(ValueError):
            SACLearner(DapPolicy(3), changed, 'rubric-v1').resume(path)
        self.path.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            SACLearner(DapPolicy(3), self.config, 'rubric-v1').resume(path)

    def test_publications_are_independent_and_periodic_retention_keeps_rollouts(self):
        run = LearnerRun(self.learner, self.root / 'run')
        first = run.export_actor()
        original = first.read_bytes()
        while self.learner.ready:
            previous = self.learner.progress['actor_updates']
            self.learner.update()
            run.after_update(previous)
        run.export_actor()
        self.assertEqual(first.read_bytes(), original)
        self.assertEqual((run.output / 'latest.pt').read_bytes(),
                         (run.output / 'published' / f'actor-{self.learner.progress["actor_updates"]:09d}.pt').read_bytes())
        self.assertEqual(len(list((run.output / 'learner').glob('checkpoint-*.pt'))), 2)
        payload = torch.load(run.output / 'latest.pt', weights_only=True)
        self.assertEqual(set(payload['models']), {'actor'})
        self.assertTrue(self.path.exists())

    def test_yaml_loads_separate_actor_and_base_encoder_sources_with_warmup(self):
        actor_path = LearnerRun(self.learner, self.root / 'actor-source').export_actor()
        base = SACLearner(DapPolicy(3), self.config, 'rubric-v1')
        base_path = LearnerRun(base, self.root / 'critic-base').export_actor()
        config = deepcopy(self.config)
        config['initialization'].update(actor_checkpoint=str(actor_path), critic_checkpoint=str(base_path))
        loaded = initialize_learner(config, 'rubric-v1', torch.device('cpu'))
        for key, value in loaded.actor.state_dict().items():
            torch.testing.assert_close(value, self.learner.actor.state_dict()[key])
        for critic in (loaded.critics.q1, loaded.critics.q2):
            for key, value in critic.image_encoder.state_dict().items():
                torch.testing.assert_close(value, base.actor.network.image_encoder.state_dict()[key])
        loaded.ingest(self.path)
        loaded.update()
        self.assertEqual(loaded.progress['warmup_updates'], 1)
        self.assertEqual(loaded.progress['actor_updates'], 0)

    def test_yaml_trained_critic_source_can_disable_warmup(self):
        self.learner.update()
        critic_path = self.learner.save(self.root / 'trained.pt')
        actor_path = LearnerRun(self.learner, self.root / 'actor-source').export_actor()
        config = deepcopy(self.config)
        config['initialization'].update(actor_checkpoint=str(actor_path), critic_checkpoint=str(critic_path), critic_is_base=False)
        config['training']['warmup'].update(enabled=False, minimum_scored_episodes=30, minimum_transitions=500)
        loaded = initialize_learner(config, 'rubric-v1', torch.device('cpu'))
        self.assertFalse(loaded.ready)
        loaded.ingest(self.path)
        loaded.update()
        loaded.update()
        self.assertEqual(loaded.progress['warmup_updates'], 0)
        self.assertEqual(loaded.progress['actor_updates'], 1)

    def test_published_policy_loads_through_existing_viewer_interface(self):
        config = load_config(CONFIG)
        learner = SACLearner(DapPolicy(3), config, 'rubric-v1')
        run = LearnerRun(learner, self.root / 'viewer-run')
        run.export_actor()
        catalog = checkpoint_catalog(self.root)
        self.assertEqual(len(catalog['items']), 1)
        self.assertFalse(catalog['warnings'])
        runner = PolicyRunner.from_checkpoint(self.root, 'viewer-run/latest.pt', 'cpu')
        self.assertEqual(runner.history_length, 3)
        self.assertTrue(all(not parameter.requires_grad for parameter in runner.policy.parameters()))
        self.assertTrue({parameter.data_ptr() for parameter in runner.policy.parameters()}.isdisjoint(
            {parameter.data_ptr() for parameter in learner.actor.parameters()}))

    @unittest.skipUnless(torch.backends.mps.is_available(), 'requires Apple MPS')
    def test_mps_updates_and_resume_optimizer_state(self):
        learner = SACLearner(DapPolicy(3).to('mps'), self.config, 'rubric-v1')
        learner.ingest(self.path)
        for _ in range(4):
            learner.update()
        path = learner.save(self.root / 'mps.pt')
        resumed = SACLearner(DapPolicy(3).to('mps'), self.config, 'rubric-v1')
        resumed.resume(path)
        resumed.update()
        self.assertEqual(next(resumed.critics.parameters()).device.type, 'mps')


class SACEntrypointTests(unittest.TestCase):
    def test_two_learners_cannot_own_the_same_run(self):
        with TemporaryDirectory() as directory:
            with training_directory(Path(directory)):
                with self.assertRaises(RuntimeError):
                    with training_directory(Path(directory)):
                        self.fail('second owner acquired the same run')

    def test_ingest_train_publish_and_resume_without_double_credits(self):
        runs = CONFIG.parents[1] / 'runs'
        runs.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=runs) as directory:
            root = Path(directory)
            config = small_config()
            config['wandb']['mode'] = 'disabled'
            source = SACLearner(DapPolicy(3), config, 'rubric-v1')
            source_path = LearnerRun(source, root / 'source').export_actor()
            config['initialization'].update(actor_checkpoint=str(source_path), critic_checkpoint=str(source_path))
            config_path = save_config(config, root / 'recipe.yaml')
            inbox = root / 'episodes'
            episode().save(inbox)
            args = argparse.Namespace(config=str(config_path), output=str(root / 'output'),
                                      episodes=str(inbox), reward_version='rubric-v1', device='cpu',
                                      watch=False, max_scored_episodes=None)
            tracking = Mock(id='synthetic-run')
            def receive_during_updates(*values, **keywords):
                if not (inbox / 'during.pt').exists():
                    episode('during').save(inbox)
            tracking.log.side_effect = receive_during_updates
            with patch('training.sac.train.start_run', return_value=tracking):
                train(args)
            checkpoint = root / 'output' / 'learner' / 'latest.pt'
            state = torch.load(checkpoint, weights_only=True)
            self.assertEqual(state['progress']['critic_updates'], 12)
            self.assertEqual(state['progress']['actor_updates'], 5)
            self.assertEqual(state['extra_state']['wandb_run_id'], 'synthetic-run')
            episode('new').save(inbox)
            config['initialization'].update(resume_checkpoint=str(checkpoint), actor_checkpoint=None, critic_checkpoint=None)
            save_config(config, config_path)
            with patch('training.sac.train.start_run', return_value=tracking) as startup:
                train(args)
                self.assertEqual(startup.call_args.kwargs['resume'], 'must')
            state = torch.load(checkpoint, weights_only=True)
            self.assertEqual(state['progress']['critic_updates'], 18)
            self.assertEqual(state['progress']['earned_updates'], 18)
            self.assertEqual(state['progress']['actor_updates'], 8)
            self.assertEqual(len(state['extra_state']['replay']), 3)
            self.assertTrue((root / 'output' / 'latest.pt').exists())


class SACConfigTests(unittest.TestCase):
    def test_base_critics_cannot_disable_warmup(self):
        config = small_config()
        config['training']['warmup']['enabled'] = False
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_invalid_settings_fail_before_training(self):
        for section, key, value in (('optimizer', 'actor_learning_rate', float('nan')),
                                    ('target_critics', 'tau', 1.), ('replay', 'replacement', False)):
            config = small_config()
            config['training'][section][key] = value
            with self.assertRaises(ValueError):
                validate_config(config)


if __name__ == '__main__':
    unittest.main()
