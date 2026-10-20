"""Finite-budget SAC optimization independent of hardware and rollout collection."""

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Iterable

from jaxtyping import Float

import torch
from torch import Tensor, nn
from torch.optim import Adam, Optimizer

from models.critic import TwinCritics
from models.policy import DapPolicy
from training.sac.config import validate_config
from training.sac.loss import actor_loss, critic_loss, critic_target, temperature_loss
from training.sac.replay import EpisodeReplay, ReplayBatch
from training.shared.checkpoints import load_checkpoint, save_checkpoint


class EntropyTemperature(nn.Module):
    """Positive entropy coefficient represented by an unconstrained log parameter."""

    def __init__(self, initial_alpha: float, device: torch.device):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(math.log(initial_alpha), device=device))

    def forward(self) -> Float[Tensor, ""]:
        return self.log_alpha.exp()


class SACLearner:
    """Own independent online/target networks, optimizer state and replay update credits."""

    def __init__(self, actor: DapPolicy, config: dict[str, Any], reward_version: str):
        validate_config(config)
        if not reward_version or actor.history_length != config['model']['history_length']:
            raise ValueError('reward version and compatible actor history are required')

        self.config = deepcopy(config)
        self.settings = self.config['training']
        self.reward_version = reward_version
        self.actor = deepcopy(actor).train().requires_grad_(True)
        self.device = next(self.actor.parameters()).device
        self.critics = TwinCritics.from_policy(self.actor).train()
        self.targets = deepcopy(self.critics).eval().requires_grad_(False)
        self.temperature = EntropyTemperature(self.settings['entropy']['initial_alpha'], self.device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.settings['optimizer']['actor_learning_rate'])
        self.critic_optimizer = Adam(self.critics.parameters(), lr=self.settings['optimizer']['critic_learning_rate'])
        self.alpha_optimizer = Adam(self.temperature.parameters(), lr=self.settings['entropy']['learning_rate'])
        self.replay = EpisodeReplay(actor.history_length, reward_version, self.settings['replay']['cache_episodes'])
        self.progress = dict(critic_updates=0, actor_updates=0, earned_updates=0, warmup_updates=0,
                             post_warmup_updates=0, imported_critics=0)
        self.wandb_run_id: str | None = None
        self.faulted = False
        self.last_metrics: dict[str, float] = {}

    @property
    def models(self):
        return dict(actor=self.actor, critics=self.critics, targets=self.targets, temperature=self.temperature)

    @property
    def optimizers(self):
        return dict(actor=self.actor_optimizer, critics=self.critic_optimizer, temperature=self.alpha_optimizer)

    def ingest(self, path: Path) -> int:
        """Earn credits exactly once for a finalized immutable scored episode."""
        count = self.replay.add(path)
        expected = (2, 3, self.config['preprocessing']['image_height'], self.config['preprocessing']['image_width'])
        if self.replay.image_shape != expected:
            self.faulted = True
            raise ValueError('episode image dimensions differ from learner preprocessing')

        self.progress['earned_updates'] += count * self.settings['critic_updates_per_new_transition']
        return count

    @property
    def ready(self) -> bool:
        if self.faulted or not len(self.replay) or self.progress['critic_updates'] >= self.progress['earned_updates']:
            return False
        if not self.settings['warmup']['enabled']:
            return True

        warmup = self.settings['warmup']
        return len(self.replay.records) >= warmup['minimum_scored_episodes'] and len(self.replay) >= warmup['minimum_transitions']

    def optimize(self, loss: Float[Tensor, ""], optimizer: Optimizer, parameters: Iterable[nn.Parameter]) -> float:
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite SAC loss; learner stopped')

        parameters = list(parameters)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(parameters, self.settings['optimizer']['max_gradient_norm'], error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in parameters):
            raise FloatingPointError('nonfinite SAC parameters; learner stopped')

        return float(norm)

    def update(self, batch: ReplayBatch | None = None) -> dict[str, float]:
        """Consume one credit; warm-up rounds never update the actor or temperature."""
        if not self.ready:
            raise RuntimeError('learner is waiting for scored data or has exhausted its budget')

        try:
            if batch is None:
                batch = self.replay.sample(self.settings['batch_size'], self.device)
            metrics = self.update_batch(batch)
        except Exception:
            self.faulted = True
            raise
        self.last_metrics = metrics
        return metrics

    def update_batch(self, batch: ReplayBatch) -> dict[str, float]:
        alpha = self.temperature().detach()
        if not torch.isfinite(alpha) or alpha <= 0:
            raise FloatingPointError('entropy coefficient must remain finite and positive')

        with torch.no_grad():
            sample = self.actor.sample(batch.next_images, batch.next_positions)
            next_values = self.targets(batch.next_images, batch.next_positions, sample.actions)
            target = critic_target(batch.rewards, batch.terminated, next_values, sample.log_prob,
                                   self.temperature(), self.settings['discount'])
        if not torch.isfinite(target).all():
            raise FloatingPointError('nonfinite Bellman target; learner stopped')

        values = self.critics(batch.images, batch.positions, batch.actions)
        loss = critic_loss(values, target)
        norm = self.optimize(loss, self.critic_optimizer, self.critics.parameters())
        with torch.no_grad():
            for target_parameter, parameter in zip(self.targets.parameters(), self.critics.parameters()):
                target_parameter.lerp_(parameter, self.settings['target_critics']['tau'])
        self.progress['critic_updates'] += 1
        metrics = {'train/critic_loss': loss.item(), 'train/critic_gradient_norm': norm,
                   'train/q1': values.q1.detach().mean().item(), 'train/q2': values.q2.detach().mean().item(),
                   'train/target_q': target.mean().item()}
        warming = self.settings['warmup']['enabled'] and self.progress['warmup_updates'] < self.settings['warmup']['critic_only_updates']
        if warming:
            self.progress['warmup_updates'] += 1
        else:
            self.progress['post_warmup_updates'] += 1
            if self.progress['post_warmup_updates'] % self.settings['critic_updates_per_actor_update'] == 0:
                metrics.update(self.update_actor(batch))
        metrics.update({'train/alpha': self.temperature().detach().item(),
                        'replay/transitions': float(len(self.replay)),
                        'train/remaining_critic_updates': float(self.progress['earned_updates'] - self.progress['critic_updates']),
                        'train/actor_updates': float(self.progress['actor_updates'])})
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError('nonfinite SAC metrics; learner stopped')

        return metrics

    def update_actor(self, batch: ReplayBatch) -> dict[str, float]:
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.critics.requires_grad_(False)
        try:
            sample = self.actor.sample(batch.images, batch.positions)
            values = self.critics(batch.images, batch.positions, sample.actions)
            loss = actor_loss(values, sample.log_prob, self.temperature())
            norm = self.optimize(loss, self.actor_optimizer, self.actor.parameters())
        finally:
            self.critics.requires_grad_(True)
        alpha_loss = temperature_loss(self.temperature.log_alpha, sample.log_prob, self.settings['entropy']['target'])
        self.optimize(alpha_loss, self.alpha_optimizer, self.temperature.parameters())
        self.progress['actor_updates'] += 1
        return {'train/actor_loss': loss.item(), 'train/actor_gradient_norm': norm,
                'train/temperature_loss': alpha_loss.item(), 'train/entropy': -sample.log_prob.detach().mean().item()}

    def save(self, path: Path) -> Path:
        """Save all resume state only after complete successful learner rounds."""
        if self.faulted:
            raise RuntimeError('refusing to overwrite a good checkpoint after a failed update')

        return save_checkpoint(path, self.models, self.optimizers, self.config, self.progress,
                               {'reward_version': self.reward_version, 'replay': self.replay.records,
                                'log_std_min': self.actor.log_std_min, 'log_std_max': self.actor.log_std_max,
                                'wandb_run_id': self.wandb_run_id})

    def resume(self, path: Path) -> None:
        """Restore exact training state and verify replay before awarding any new credits."""
        payload = torch.load(path, weights_only=True, map_location='cpu')
        saved_recipe = {key: value for key, value in payload['config'].items() if key != 'initialization'}
        current_recipe = {key: value for key, value in self.config.items() if key != 'initialization'}
        if saved_recipe != current_recipe or payload['extra_state']['reward_version'] != self.reward_version:
            raise ValueError('resume requires identical config and reward definition')

        if (payload['extra_state']['log_std_min'], payload['extra_state']['log_std_max']) != (self.actor.log_std_min, self.actor.log_std_max):
            raise ValueError('resume actor distribution bounds differ')

        replay = EpisodeReplay(self.actor.history_length, self.reward_version, self.settings['replay']['cache_episodes'])
        for record in payload['extra_state']['replay']:
            replay.add(Path(record['path']), record['sha256'])
        state = load_checkpoint(path, self.models, self.optimizers)
        self.wandb_run_id = state.extra_state['wandb_run_id']
        self.progress = state.progress
        self.replay = replay
        self.targets.eval().requires_grad_(False)

    def import_critics(self, path: Path) -> None:
        """Import trained compatible critics; the YAML explicitly controls warmup."""
        if self.progress['critic_updates'] or len(self.replay):
            raise ValueError('import critics only before ingestion and learning')

        payload = torch.load(path, weights_only=True, map_location='cpu')
        if payload['extra_state']['reward_version'] != self.reward_version or payload['progress']['critic_updates'] < 1:
            raise ValueError('critic import requires trained values under the same reward definition')

        for key in ('model', 'preprocessing'):
            if payload['config'][key] != self.config[key]:
                raise ValueError('critic checkpoint observation contract differs')

        if payload['config']['training']['discount'] != self.settings['discount'] or payload['config']['training']['entropy'] != self.settings['entropy']:
            raise ValueError('critic soft-return definition differs')

        if any(not torch.isfinite(value).all() for value in payload['models']['critics'].values()):
            raise ValueError('critic checkpoint contains nonfinite weights')

        self.critics.load_state_dict(payload['models']['critics'])
        self.targets.load_state_dict(self.critics.state_dict())
        self.progress['imported_critics'] = 1
