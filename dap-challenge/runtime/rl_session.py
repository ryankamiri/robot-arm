"""Manage persisted RL runs independently from the arm connection and SAC learner process."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import RLock

import torch

from models.policy import DapPolicy
from recording.rl_store import RolloutStore, safe_identifier
from runtime.policy import PolicyRunner
from runtime.session import RuntimeSession
from shared.config import load_config, save_config, save_json
from shared.policy_snapshots import encode_actor
from shared.rewards import load_rubric
from training.sac.config import validate_config
from training.sac.run import load_actor
from training.shared.checkpoints import save_checkpoint


class RLSession:
    def __init__(self, runtime: RuntimeSession, project: Path, demo: bool, settings: dict | None = None):
        self.runtime, self.project, self.demo = runtime, project, demo
        self.settings = settings or load_config(project / 'configs/rl.yaml')
        self.root = project / 'rollouts' / ('.demo' if demo else '')
        self.runs = project / 'runs'
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = self.owner = self.store = self.process = self.output = None
        self.lock = RLock()

    def catalog(self) -> list[str]:
        return sorted(path.parent.name for path in self.root.glob('*/session.json'))

    def require_owner(self, owner: str) -> None:
        if not self.run_id or self.owner != owner:
            raise ValueError('Load an RL run in this browser first.')

    def alive(self, output: Path) -> bool:
        receipt = output / 'launch.json'
        if not receipt.exists():
            return False

        pid = json.loads(receipt.read_text())['pid']
        result = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True)
        return result.returncode == 0 and 'training.sac.train' in result.stdout and str(output) in result.stdout

    def load_run(self, owner: str, run_id: str, publication_url: str) -> dict:
        with self.lock:
            if self.process is not None:
                self.process.poll()
            if self.output is not None and self.alive(self.output):
                raise ValueError('Stop training before loading another run.')

            if self.runtime.collector is not None or (self.runtime.last_collector and self.runtime.last_collector.thread.is_alive()):
                raise ValueError('End the rollout and wait for its recording to finish before switching runs.')

            run_id = safe_identifier(run_id)
            if self.demo and not run_id.startswith('demo-'):
                run_id = 'demo-' + run_id
            if not self.demo and run_id.startswith('demo-'):
                raise ValueError('Demo runs cannot become physical RL runs.')

            output = self.runs / run_id
            directory = self.root / run_id
            if output.is_symlink() or directory.is_symlink() or self.alive(output):
                raise ValueError('This run is already owned by a learner or is not a local run directory.')

            manifest_path = directory / 'session.json'
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                config = manifest['learner_config']
                if manifest['demo'] != self.demo:
                    raise ValueError('Run hardware mode differs from this server.')

                rubric = manifest['rubric']
                if (output / 'learner/latest.pt').exists():
                    config['initialization']['resume_checkpoint'] = str(output / 'learner/latest.pt')
            else:
                if output.exists():
                    raise ValueError('Choose a fresh RL run name; its checkpoint directory already exists.')

                config = load_config(self.project.parent / self.settings['learner']['config'])
                rubric = load_rubric(self.project.parent / self.settings['reward']['config'])
                manifest = {'run_id': run_id, 'demo': self.demo, 'created_at': datetime.now(timezone.utc).isoformat(),
                            'rubric': rubric, 'maximum': self.settings['session']['max_scored_episodes']}
                output.mkdir(parents=True)
                if self.demo:
                    config['wandb']['mode'] = 'disabled'
                    base = output / 'demo-base.pt'
                    save_checkpoint(base, {'actor': DapPolicy(**config['model'])}, {}, config, {'optimizer_step': 0})
                    config['initialization'].update(actor_checkpoint=str(base), critic_checkpoint=str(base), critic_is_base=True, resume_checkpoint=None)
                config['wandb']['name'] = run_id
            config = deepcopy(config)
            config['training']['publication']['url'] = publication_url
            validate_config(config)
            source = config['initialization']['resume_checkpoint'] or config['initialization']['actor_checkpoint']
            source = str(self.project.parent / source) if not Path(source).is_absolute() else source
            actor = load_actor(Path(source), config, torch.device('cpu'))
            step = 0
            if config['initialization']['resume_checkpoint']:
                checkpoint = torch.load(source, map_location='cpu', weights_only=True)
                step = checkpoint['progress']['actor_updates']
            snapshot = encode_actor(actor, config['preprocessing'], run_id, step)
            runner = PolicyRunner.from_snapshot(snapshot.content)
            store = RolloutStore(directory, rubric)

            self.runtime.request('rl_reserve', {'client_id': owner, 'run_id': run_id, 'runner': runner, 'store': store})
            self.run_id, self.owner, self.output, self.store = run_id, owner, output, store
            self.process = None
            manifest['learner_config'] = config
            save_json(manifest, manifest_path)
            save_config(config, output / 'session-config.yaml')
            save_config(rubric, output / 'reward.yaml')
            return self.status(owner)

    def start_training(self, owner: str) -> dict:
        with self.lock:
            self.require_owner(owner)
            if self.alive(self.output):
                raise ValueError('Training is already running for this run.')

            manifest_path = self.store.root / 'session.json'
            manifest = json.loads(manifest_path.read_text())
            config = deepcopy(manifest['learner_config'])
            checkpoint = self.output / 'learner/latest.pt'
            if checkpoint.exists():
                config['initialization']['resume_checkpoint'] = str(checkpoint)
            validate_config(config)
            manifest['learner_config'] = config
            save_json(manifest, manifest_path)
            save_config(config, self.output / 'session-config.yaml')
            for name in ('drain.request', 'stop.request', 'status.json', 'progress.json'):
                (self.output / name).unlink(missing_ok=True)
            command = [sys.executable, '-m', 'training.sac.train', '--config', str(self.output / 'session-config.yaml'),
                       '--output', str(self.output), '--episodes', str(self.store.root / 'scored'),
                       '--reward-version', self.store.rubric['version'],
                       '--device', 'cpu' if self.demo else self.settings['session']['device'], '--watch',
                       '--max-scored-episodes', str(manifest['maximum']), '--owner-pid', str(os.getpid()),
                       '--stop-file', str(self.output / 'stop.request')]
            environment = {**os.environ, 'PYTHONPATH': str(self.project), 'PYTHONUNBUFFERED': '1'}
            try:
                with (self.output / 'learner.log').open('a') as log:
                    self.process = subprocess.Popen(command, cwd=self.project.parent, env=environment, stdout=log, stderr=subprocess.STDOUT)
                save_json({'pid': self.process.pid, 'command': command, 'demo': self.demo}, self.output / 'launch.json')
            except Exception as error:
                save_json({'state': 'failed', 'error': str(error)}, self.output / 'status.json')
                raise
            return self.status(owner)

    def stop_training(self, owner: str) -> dict:
        with self.lock:
            self.require_owner(owner)
            if self.alive(self.output):
                (self.output / 'stop.request').touch()
            return self.status(owner)

    def status(self, owner: str) -> dict:
        with self.lock:
            output = self.output
            learner = {'state': 'not_started'}
            if output is not None:
                for name in ('progress.json', 'status.json'):
                    if (output / name).exists():
                        learner = json.loads((output / name).read_text())
                alive = self.alive(output)
                learner['alive'] = alive
                if self.process is not None and self.process.poll() is not None:
                    learner['exit_code'] = self.process.returncode
                    if self.process.returncode != 0:
                        learner.update(state='failed', error=(output / 'learner.log').read_text()[-4000:])
                elif alive and learner['state'] == 'not_started':
                    learner['state'] = 'starting'
                if (output / 'stop.request').exists() and alive:
                    learner['state'] = 'stopping'
                elif (output / 'drain.request').exists() and alive:
                    learner['state'] = 'draining'
            mine = self.owner == owner
            return {'run_id': self.run_id, 'last_run_id': output.name if output else None, 'owned_by_you': mine, 'demo': self.demo, 'runs': self.catalog(),
                    'learner': learner, 'rubric': self.store.rubric if self.store else None,
                    'episodes': self.store.episodes() if self.store else [],
                    'maximum': json.loads((self.store.root / 'session.json').read_text())['maximum'] if self.store else None,
                    'active_attempt': self.runtime.collector.metadata['episode_id'] if self.runtime.collector else None}

    def start_attempt(self, owner: str, tasks: list[str]) -> dict:
        with self.lock:
            self.require_owner(owner)
            state = self.status(owner)
            if sum(episode['status'] == 'scored' for episode in state['episodes']) >= state['maximum']:
                raise ValueError('This run has reached its scored-attempt budget.')

            if any(episode['status'] in ('recording', 'review') for episode in state['episodes']):
                raise ValueError('Score or abort the pending attempt before collecting another.')

            self.runtime.request('start', {'client_id': owner, 'mode': 'sample', 'rl_attempt': True, 'tasks': tasks})
            return self.status(owner)

    def stop_attempt(self, owner: str, abort: bool = False) -> dict:
        with self.lock:
            self.require_owner(owner)
            self.runtime.request('abort' if abort else 'stop', {'client_id': owner})
            return self.status(owner)

    def score(self, owner: str, episode_id: str, cued: list[str], answers: dict, notes: str) -> dict:
        with self.lock:
            self.require_owner(owner)
            self.store.score(episode_id, cued, answers, notes)
            return self.status(owner)

    def discard(self, owner: str, episode_id: str) -> dict:
        with self.lock:
            self.require_owner(owner)
            path = self.store.path(episode_id)
            metadata = json.loads((path / 'metadata.json').read_text())
            if metadata['status'] != 'review':
                raise ValueError('Only an unscored reviewed attempt can be aborted.')

            metadata.update(status='aborted', error='Discarded during human review.')
            write_path = path / 'metadata.json'
            save_json(metadata, write_path)
            path.rename(self.store.root / 'aborted' / path.name)
            return self.status(owner)

    def end_session(self, owner: str) -> dict:
        with self.lock:
            self.require_owner(owner)
            if self.runtime.devices is not None:
                self.runtime.request('disconnect', {'client_id': owner})
            return self.status(owner)

    def close(self) -> None:
        if self.run_id is not None:
            self.runtime.request('rl_release', {'client_id': self.owner})
        writer = self.runtime.last_collector
        if writer is not None:
            writer.thread.join(timeout=15)
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                raise TimeoutError('SAC learner is still saving; do not start another learner for this run.')
