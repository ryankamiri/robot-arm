"""SAC run persistence and immutable actor publications compatible with the viewer."""

from copy import deepcopy
from pathlib import Path

import torch

from models.policy import DapPolicy
from shared.policy_snapshots import encode_actor, send_actor
from training.sac.learner import SACLearner
from training.shared.checkpoints import save_checkpoint
from training.shared.utils import atomic_copy


def load_actor(path: Path, config: dict, device: torch.device) -> DapPolicy:
    """Read BC, published actor or full SAC state without connecting any hardware."""
    state = torch.load(path, weights_only=True, map_location='cpu')
    if state.get('format_version') != 1 or state['config']['preprocessing'] != config['preprocessing']:
        raise ValueError('actor checkpoint preprocessing is incompatible')

    if state['config']['model']['history_length'] != config['model']['history_length']:
        raise ValueError('actor checkpoint history is incompatible')

    actor = DapPolicy(**config['model']).to(device)
    source_bounds = (state['config']['model'].get('log_std_min', -5.), state['config']['model'].get('log_std_max', 2.))
    if source_bounds != (actor.log_std_min, actor.log_std_max):
        raise ValueError('actor checkpoint sampling bounds differ from learner config')

    actor.load_state_dict(state['models']['actor'])
    if any(not torch.isfinite(parameter).all() for parameter in actor.parameters()):
        raise ValueError('actor checkpoint has nonfinite weights')

    return actor


def initialize_learner(config: dict, reward_version: str, device: torch.device) -> SACLearner:
    """Load separate YAML actor/critic sources or restore the complete learner state."""
    initialization = config['initialization']
    resume = initialization['resume_checkpoint']
    actor = load_actor(Path(resume or initialization['actor_checkpoint']), config, device)
    learner = SACLearner(actor, config, reward_version)
    if resume:
        learner.resume(Path(resume))
    elif initialization['critic_is_base']:
        base = load_actor(Path(initialization['critic_checkpoint']), config, device)
        for critic in (learner.critics.q1, learner.critics.q2):
            critic.initialize_from_policy(base)
        learner.targets.load_state_dict(learner.critics.state_dict())
    else:
        learner.import_critics(Path(initialization['critic_checkpoint']))
    return learner


class LearnerRun:
    """Keep resumable learner files separate from inference-only published policies."""

    def __init__(self, learner: SACLearner, output: Path):
        self.learner = learner
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.publication_ack: dict | None = None

    def save(self, periodic: bool = False) -> None:
        self.learner.save(self.output / 'learner' / 'latest.pt')
        self.export_actor()
        if periodic:
            step = self.learner.progress['critic_updates']
            self.learner.save(self.output / 'learner' / f'checkpoint-{step:09d}.pt')
            paths = sorted((self.output / 'learner').glob('checkpoint-*.pt'))
            for path in paths[:-self.learner.settings['checkpoint']['keep_periodic']]:
                path.unlink()

    def export_actor(self) -> Path:
        """Persist an actor alongside recovery checkpoints, independently of live transfers."""
        if self.learner.faulted:
            raise RuntimeError('cannot publish after a failed learner update')

        step = self.learner.progress['actor_updates']
        version = f'{self.output.name}/actor-{step:09d}'
        path = self.output / 'published' / f'actor-{step:09d}.pt'
        extra = {'policy_version': version, 'reward_version': self.learner.reward_version,
                 'critic_updates': self.learner.progress['critic_updates']}
        config = deepcopy(self.learner.config)
        config['model'] = {'history_length': self.learner.actor.history_length,
                           'log_std_min': self.learner.actor.log_std_min,
                           'log_std_max': self.learner.actor.log_std_max}
        if not path.exists():
            save_checkpoint(path, {'actor': self.learner.actor}, {}, config, {'optimizer_step': step}, extra)
        published = torch.load(path, weights_only=True, map_location='cpu')
        for key, value in self.learner.actor.state_dict().items():
            if not torch.equal(value.detach().cpu(), published['models']['actor'][key]):
                raise ValueError('publication version collision; choose a new run directory')

        # latest.pt is an atomic discovery pointer encoded as the same actor-only archive.
        atomic_copy(path, self.output / 'latest.pt')
        return path

    def publish(self) -> dict:
        """Send a cloned actor directly to FastAPI without writing or reading a snapshot file."""
        if self.learner.faulted:
            raise RuntimeError('cannot publish after a failed learner update')

        settings = self.learner.settings['publication']
        if not settings['enabled']:
            return {'publication_state': 'disabled'}

        snapshot = encode_actor(self.learner.actor, self.learner.config['preprocessing'],
                                self.output.name, self.learner.progress['actor_updates'])
        try:
            self.publication_ack = send_actor(snapshot, settings['url'], settings['timeout_s'])
        except Exception:
            self.save()
            raise
        return self.publication_ack

    def after_update(self, previous_actor_updates: int) -> None:
        progress = self.learner.progress
        if progress['actor_updates'] != previous_actor_updates and progress['actor_updates'] % self.learner.settings['publication']['every_actor_updates'] == 0:
            self.publish()
        if progress['critic_updates'] % self.learner.settings['checkpoint']['every_critic_updates'] == 0:
            self.save(periodic=True)
