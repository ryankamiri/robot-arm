"""Exercise human grading, exact RL transition boundaries, recovery and background session ownership."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch
from uuid import uuid4
from types import SimpleNamespace

import av
import torch

from recording.rl_store import RolloutStore
from runtime.rl_session import RLSession
from runtime.session import RuntimeSession
from shared.config import load_config, save_config, save_json
from shared.demo_devices import DemoFollower
from shared.ownership import HardwareLease
from shared.rewards import load_rubric, score_answers
from test_runtime import FixedRunner
from training.sac.replay import EpisodeReplay, ScoredEpisode


PROJECT = Path(__file__).parents[1].resolve()


def wait_until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.03)
    raise AssertionError('Timed out waiting for asynchronous completion.')


class RubricTests(unittest.TestCase):
    def setUp(self):
        self.rubric = load_rubric(PROJECT / 'configs/reward.yaml')
        self.bump = {'contact': True, 'retract': 1, 'clean': 1, 'tracking': None}
        self.hi = {'response': True, 'mouth': 1, 'up': 1, 'down': 1, 'tracking': None}

    def test_gates_bonus_na_and_multi_task_average(self):
        result = score_answers(self.rubric, ['fist_bump'], ['fist_bump'], {'fist_bump': self.bump})
        self.assertAlmostEqual(result['reward'], .9)
        self.bump['tracking'] = 1
        self.assertAlmostEqual(score_answers(self.rubric, ['fist_bump'], ['fist_bump'], {'fist_bump': self.bump})['reward'], 1.)
        self.bump['contact'] = False
        self.assertEqual(score_answers(self.rubric, ['fist_bump'], ['fist_bump'], {'fist_bump': self.bump})['reward'], 0)
        self.assertAlmostEqual(score_answers(self.rubric, ['fist_bump', 'greeting'], ['fist_bump', 'greeting'], {'fist_bump': self.bump, 'greeting': self.hi})['reward'], .45)
        self.assertAlmostEqual(score_answers(self.rubric, ['fist_bump', 'greeting'], ['greeting'], {'greeting': self.hi})['reward'], .9)

    def test_rejects_unknown_missing_nonfinite_and_unselected_answers(self):
        for selected, cued, answers in [([], [], {}), (['greeting'], ['fist_bump'], {'fist_bump': self.bump}),
                                       (['greeting'], [], {}), (['greeting'], ['greeting'], {'greeting': {}}),
                                       (['greeting'], ['greeting'], {'greeting': {**self.hi, 'up': float('nan')}}),
                                       (['greeting'], ['greeting'], {'greeting': {**self.hi, 'response': 1}})]:
            with self.assertRaises(ValueError):
                score_answers(self.rubric, selected, cued, answers)

    def test_multiple_gates_require_every_gate_and_keep_binary_answers(self):
        rubric = deepcopy(self.rubric)
        task = rubric['tasks']['fist_bump']
        task['gates'] = ['contact', 'retract']
        answers = {**self.bump, 'retract': True}
        for contact, retract, expected in [(True, True, .9), (True, False, 0), (False, True, 0), (False, False, 0)]:
            answers.update(contact=contact, retract=retract)
            result = score_answers(rubric, ['fist_bump'], ['fist_bump'], {'fist_bump': answers})
            self.assertAlmostEqual(result['reward'], expected)
        answers.update(contact=True, retract=.5)
        with self.assertRaises(ValueError):
            score_answers(rubric, ['fist_bump'], ['fist_bump'], {'fist_bump': answers})

    def test_gate_configuration_rejects_missing_duplicate_unknown_and_optional_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'rubric.yaml'
            for gates in ([], 'contact', ['contact', 'contact'], ['unknown'], ['tracking']):
                config = load_config(PROJECT / 'configs/reward.yaml')
                config['tasks']['fist_bump']['gates'] = gates
                save_config(config, path)
                with self.subTest(gates=gates), self.assertRaises(ValueError):
                    load_rubric(path)
            config['tasks']['fist_bump']['gates'] = ['contact', 'retract']
            save_config(config, path)
            self.assertEqual(load_rubric(path)['tasks']['fist_bump']['gates'], ['contact', 'retract'])

    def test_version_changes_with_definition(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'rubric.yaml'
            source = load_config(PROJECT / 'configs/reward.yaml')
            source['tasks']['greeting']['label'] = 'Changed rubric'
            save_config(source, path)
            self.assertNotEqual(load_rubric(path)['version'], self.rubric['version'])


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rubric = load_rubric(PROJECT / 'configs/reward.yaml')
        self.store = RolloutStore(self.root / 'rollouts/test', self.rubric)
        config = load_config(PROJECT / 'configs/runtime.yaml')
        config['policy']['device'] = 'cpu'
        self.runtime = RuntimeSession(config, self.root / 'pose.yaml', HardwareLease(self.root / 'lease'), lambda settings: DemoFollower())
        self.runner = FixedRunner()
        self.owner = {'client_id': 'test'}
        self.runtime.handle_command('rl_reserve', {**self.owner, 'run_id': 'test', 'runner': self.runner, 'store': self.store})
        self.runtime.handle_command('connect', {**self.owner, 'settings': {}})
        self.runtime.handle_command('capture_pose', self.owner)
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.runtime.disconnect()
        if self.runtime.last_collector:
            self.runtime.last_collector.thread.join(timeout=5)

    def start_attempt(self):
        self.runtime.handle_command('start', {**self.owner, 'mode': 'sample', 'rl_attempt': True, 'tasks': ['fist_bump']})
        return self.runtime.collector

    def collect(self):
        writer = self.start_attempt()
        self.runtime.tick(time.monotonic(), 1 / 30)
        self.runtime.tick(time.monotonic(), 1 / 30)
        self.runtime.handle_command('stop', self.owner)
        writer.thread.join(timeout=10)
        self.assertIsNone(writer.error)
        return writer

    def test_exact_sent_targets_final_observation_and_terminal_replay(self):
        writer = self.collect()
        episode_id = writer.metadata['episode_id']
        path = self.store.path(episode_id)
        rows = [json.loads(line) for line in (path / 'steps.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]['positions'], [0.] * 5 + [50.])
        self.assertEqual(rows[0]['requested'], [80.] * 5 + [100.])
        self.assertEqual(rows[0]['sent'], [2.] * 5 + [52.])
        self.assertTrue(rows[-1]['final'])
        self.assertNotIn('requested', rows[-1])
        self.assertEqual(rows[-1]['positions'], [4.] * 5 + [54.])
        self.assertEqual(list((self.store.root / 'scored').glob('*.pt')), [])
        for name in ('video.mp4', 'mac_video.mp4'):
            with av.open(str(path / name)) as video:
                self.assertEqual(len(list(video.decode(video=0))), 3)
        answers = {'fist_bump': {'contact': True, 'retract': 1, 'clean': 1, 'tracking': None}}
        metadata = self.store.score(episode_id, ['fist_bump'], answers, 'synthetic')
        self.assertAlmostEqual(metadata['reward'], .9)
        scored = self.store.root / 'scored' / f'{episode_id}.pt'
        episode = ScoredEpisode.load(scored)
        self.assertEqual(episode.images.shape, (3, 2, 3, 168, 224))
        self.assertEqual(episode.policy_version, self.runner.version)
        replay = EpisodeReplay(3, self.rubric['version'], 1)
        replay.add(scored)
        batch = replay.sample(64, torch.device('cpu'))
        self.assertTrue(torch.equal(batch.rewards != 0, batch.terminated))
        self.assertTrue(torch.allclose(batch.next_positions[batch.terminated, -1], episode.positions[-1].expand(batch.terminated.sum(), -1)))
        self.store.score(episode_id, ['fist_bump'], answers, 'synthetic')
        self.assertEqual(len(list((self.store.root / 'scored').glob('*.pt'))), 1)
        with self.assertRaisesRegex(ValueError, 'immutable'):
            self.store.score(episode_id, ['fist_bump'], answers, 'changed')

    def test_fault_quarantines_attempt_and_cannot_be_scored(self):
        writer = self.start_attempt()
        self.runtime.tick(time.monotonic(), 1 / 30)
        self.runtime.handle_fault(RuntimeError('camera lost'))
        writer.thread.join(timeout=10)
        metadata = self.store.episodes()[0]
        self.assertEqual(metadata['status'], 'aborted')
        self.assertIn('camera lost', metadata['error'])
        with self.assertRaisesRegex(ValueError, 'finalized'):
            self.store.score(metadata['episode_id'], [], {}, '')
        self.assertFalse(list((self.store.root / 'scored').glob('*.pt')))

    def test_rl_reservation_blocks_unrecorded_start_load_and_reset(self):
        with self.assertRaisesRegex(ValueError, 'recorded'):
            self.runtime.handle_command('start', {**self.owner, 'mode': 'mean'})
        with self.assertRaisesRegex(ValueError, 'owns'):
            self.runtime.handle_command('load', {**self.owner, 'runner': self.runner})
        writer = self.start_attempt()
        with self.assertRaisesRegex(ValueError, 'Stop and review'):
            self.runtime.handle_command('reset', self.owner)
        self.runtime.handle_command('abort', self.owner)
        writer.thread.join(timeout=10)

    def test_new_actor_stays_pending_until_the_next_attempt(self):
        writer = self.start_attempt()
        newer = FixedRunner()
        newer.version = 'test/actor-next'
        newer.publication = SimpleNamespace(run_id='test', actor_updates=1, digest='test-digest')
        self.runtime.stage_publication(newer)
        self.assertIs(self.runtime.runner, self.runner)
        self.runtime.tick(time.monotonic(), 1 / 30)
        self.runtime.handle_command('stop', self.owner)
        writer.thread.join(timeout=10)
        self.assertEqual(writer.metadata['policy_version'], self.runner.version)
        next_writer = self.start_attempt()
        self.assertIs(self.runtime.runner, newer)
        self.assertEqual(next_writer.metadata['policy_version'], newer.version)
        self.runtime.handle_command('abort', self.owner)
        next_writer.thread.join(timeout=10)

    def test_recover_crashed_pending_and_finish_scoring_after_commit_interruption(self):
        directory = self.store.root / 'pending' / 'interrupted'
        directory.mkdir()
        save_json({'episode_id': 'interrupted', 'status': 'recording'}, directory / 'metadata.json')
        recovered = RolloutStore(self.store.root, self.rubric)
        self.assertEqual(json.loads((recovered.path('interrupted') / 'metadata.json').read_text())['status'], 'aborted')
        writer = self.collect()
        answers = {'fist_bump': {'contact': False, 'retract': 0, 'clean': 0, 'tracking': None}}
        with patch.object(self.store, 'commit_score', side_effect=OSError('interrupted rename')):
            with self.assertRaises(OSError):
                self.store.score(writer.metadata['episode_id'], ['fist_bump'], answers, '')
        recovered = RolloutStore(self.store.root, self.rubric)
        self.assertTrue((recovered.root / 'scored' / f"{writer.metadata['episode_id']}.pt").is_file())


class SessionProcessTests(unittest.TestCase):
    def test_synthetic_background_learner_drains_and_resumes_without_recollection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime_config = load_config(PROJECT / 'configs/runtime.yaml')
            runtime_config['policy']['device'] = 'cpu'
            runtime_config['control']['consecutive_deadline_misses'] = 100
            runtime_config['control']['browser_timeout_s'] = 60
            runtime = RuntimeSession(runtime_config, root / 'pose.yaml', HardwareLease(root / 'lease'), lambda settings: DemoFollower())
            config = load_config(PROJECT / 'configs/sac.yaml')
            config['training']['publication']['enabled'] = False
            config['training']['batch_size'] = 1
            config['training']['warmup'].update(minimum_scored_episodes=1, minimum_transitions=1, critic_only_updates=2)
            config_path = save_config(config, root / 'sac.yaml')
            settings = {'session': {'max_scored_episodes': 50, 'device': 'cpu'}, 'learner': {'config': str(config_path)}, 'reward': {'config': str(PROJECT / 'configs/reward.yaml')}}
            session = RLSession(runtime, PROJECT, True, settings)
            session.root = root / 'rollouts'
            session.root.mkdir()
            name = 'demo-test-' + uuid4().hex
            output = PROJECT / 'runs' / name
            runtime.start()
            try:
                state = session.start('owner', name, 'http://127.0.0.1:9999/api/inference/policy')
                self.assertTrue(state['learner']['alive'])
                wait_until(lambda: (output / 'learner/latest.pt').is_file(), 30)
                runtime.request('connect', {'client_id': 'owner', 'settings': {}})
                runtime.request('capture_pose', {'client_id': 'owner'})
                session.start_attempt('owner', ['greeting'])
                wait_until(lambda: runtime.state['requested_target'] is not None, 10)
                session.stop_attempt('owner')
                wait_until(lambda: any(item['status'] == 'review' for item in session.store.episodes()))
                episode = session.store.episodes()[0]
                session.score('owner', episode['episode_id'], ['greeting'], {'greeting': {'response': True, 'mouth': 1, 'up': 1, 'down': 1, 'tracking': None}}, '')
                session.end('owner')
                self.assertIsNone(runtime.devices)
                session.process.wait(timeout=90)
                terminal = json.loads((output / 'status.json').read_text())
                self.assertEqual(session.process.returncode, 0, (output / 'learner.log').read_text()[-3000:])
                self.assertEqual(terminal['critic_updates'], terminal['earned_updates'])
                self.assertGreater(terminal['critic_updates'], 0)
                session.start('owner', name, 'http://127.0.0.1:9998/api/inference/policy')
                wait_until(lambda: (output / 'progress.json').exists(), 30)
                session.end('owner')
                session.process.wait(timeout=60)
                resumed = json.loads((output / 'status.json').read_text())
                self.assertEqual(resumed['critic_updates'], terminal['critic_updates'])
                self.assertEqual(resumed['earned_updates'], terminal['earned_updates'])
            finally:
                session.close()
                runtime.close()
                shutil.rmtree(output, ignore_errors=True)
