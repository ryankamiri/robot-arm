"""Behavior cloning entry point with held-out validation and latest/best checkpoints."""

import argparse
import math
from pathlib import Path
from typing import Any, Mapping

from jaxtyping import Float, UInt8
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from models.policy import DapPolicy
from training.bc.dataset import BCSample, create_dataloaders
from training.bc.loss import expert_nll
from training.shared.checkpoints import load_checkpoint, save_checkpoint
from training.shared.config import load_config, save_config
from training.shared.episodes import discover_episodes
from training.shared.metrics import log_metrics, start_run
from training.shared.randomness import seed_training


def batch_nll(
    policy: DapPolicy, batch: Mapping[str, Any], device: torch.device, epsilon: float,
) -> Float[Tensor, 'B']:
    """Move observations and expert targets to the chosen float32 training device."""
    images: Float[Tensor, 'B L I C_in H_in W_in'] = batch['images'].to(device=device, dtype=torch.float32)
    positions: Float[Tensor, 'B L A'] = batch['positions'].to(device=device, dtype=torch.float32)
    targets: Float[Tensor, 'B A'] = batch['expert_action'].to(device=device, dtype=torch.float32)
    return expert_nll(policy(images, positions), targets, epsilon)


def evaluate(
    policy: DapPolicy, loader: DataLoader[BCSample], device: torch.device, epsilon: float,
) -> float:
    """Weight every held-out window equally and restore the policy's previous mode."""
    previous_mode = policy.training
    total, count = 0.0, 0
    policy.eval()
    try:
        with torch.inference_mode():
            for batch in loader:
                losses: Float[Tensor, 'B'] = batch_nll(policy, batch, device, epsilon)
                total += losses.sum().item()
                count += losses.numel()
        if count == 0 or not math.isfinite(total):
            raise ValueError('validation must contain samples and have finite NLL')

        return total / count
    finally:
        policy.train(previous_mode)


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject invalid training settings before creating a run or updating weights."""
    training = config['training']
    for value in (training['max_epochs'], training['evaluation']['every_steps'], training['checkpoint']['every_steps']):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('epoch counts and evaluation/checkpoint intervals must be positive integers')
    if not isinstance(training['evaluation']['at_end'], bool):
        raise ValueError('evaluation.at_end must be boolean')
    for key in ('learning_rate', 'weight_decay', 'max_gradient_norm'):
        value = training['optimizer'][key]
        if isinstance(value, bool) or not math.isfinite(value) or value < 0 or (key != 'weight_decay' and value == 0):
            raise ValueError(f'invalid optimizer setting: {key}')
    epsilon = training['loss']['endpoint_epsilon']
    if isinstance(epsilon, bool) or not math.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError('endpoint_epsilon must lie strictly between zero and one')


def train(
    config: dict[str, Any], output: str | Path, device: str,
    wandb_project: str | None = None, wandb_mode: str | None = None, resume: str | Path | None = None,
) -> Path:
    """Train locally; explicit output/device/tracking settings avoid hidden run defaults."""
    validate_config(config)
    tracking = config.get('wandb', {})
    wandb_project = wandb_project or tracking.get('project')
    wandb_mode = wandb_mode or tracking.get('mode')
    if not wandb_project or not wandb_mode:
        raise ValueError('provide W&B project and mode in config or CLI')

    output = Path(output).resolve()
    target_device = torch.device(device)
    settings = config['training']
    seed_training(settings['seed'])
    loaders = create_dataloaders(discover_episodes(config['dataset']['root']), config, settings['seed'])
    policy = DapPolicy(**config['model']).to(device=target_device, dtype=torch.float32)
    optimizer = AdamW(policy.parameters(), lr=settings['optimizer']['learning_rate'],
                      weight_decay=settings['optimizer']['weight_decay'])
    split = {'train': list(loaders.train_episode_ids), 'validation': list(loaders.validation_episode_ids)}
    epoch, next_batch, step, best_nll = 0, 0, 0, math.inf
    epoch_generator_state: UInt8[Tensor, 'N_rng'] = loaders.generator.get_state()
    previous_run_id = None
    if resume is not None:
        if not output.is_dir():
            raise ValueError('resume requires the existing run directory')

        state = load_checkpoint(resume, {'actor': policy}, {'actor': optimizer})
        if state.config != config or state.extra_state['split'] != split:
            raise ValueError('resume requires the same config and rollout split')

        epoch, next_batch, step = (state.progress[key] for key in ('epoch', 'next_batch', 'optimizer_step'))
        best_nll = state.extra_state['best_validation_nll']
        epoch_generator_state = state.extra_state['epoch_generator_state']
        previous_run_id = state.extra_state['wandb_run_id']
    else:
        output.mkdir(parents=True, exist_ok=False)
        save_config(config, output / 'config.yaml')

    epsilon = settings['loss']['endpoint_epsilon']
    run = start_run(wandb_project, config, wandb_mode, output, run_id=previous_run_id,
                    resume='must' if previous_run_id is not None and wandb_mode == 'online' else None,
                    entity=tracking.get('entity'), name=tracking.get('name'))
    last_evaluation_step = None

    def checkpoint(path: Path) -> None:
        save_checkpoint(path, {'actor': policy}, {'actor': optimizer}, config,
                        {'epoch': epoch, 'next_batch': next_batch, 'optimizer_step': step},
                        {'split': split, 'best_validation_nll': best_nll,
                         'epoch_generator_state': epoch_generator_state, 'wandb_run_id': run.id})

    def validation() -> float:
        nonlocal best_nll, last_evaluation_step
        value = evaluate(policy, loaders.validation, target_device, epsilon)
        if value < best_nll:
            best_nll = value
            checkpoint(output / 'best.pt')
        last_evaluation_step = step
        return value

    try:
        policy.train()
        while epoch < settings['max_epochs']:
            loaders.generator.set_state(epoch_generator_state)
            iterator = iter(loaders.train)
            for _ in range(next_batch):
                next(iterator)
            for batch in iterator:
                optimizer.zero_grad(set_to_none=True)
                loss: Float[Tensor, ''] = batch_nll(policy, batch, target_device, epsilon).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('non-finite training loss; no update performed')

                loss.backward()
                gradient_norm: Float[Tensor, ''] = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), settings['optimizer']['max_gradient_norm'], error_if_nonfinite=True,
                )
                optimizer.step()
                step += 1
                next_batch += 1
                metrics = {'train/nll': loss.item(), 'train/gradient_norm': gradient_norm.item(), 'train/epoch': epoch}
                if step % settings['evaluation']['every_steps'] == 0:
                    metrics['val/nll'] = validation()
                if step % settings['checkpoint']['every_steps'] == 0:
                    checkpoint(output / 'latest.pt')
                log_metrics(run, metrics, step)
            epoch += 1
            next_batch = 0
            epoch_generator_state = loaders.generator.get_state()
        if settings['evaluation']['at_end'] and last_evaluation_step != step:
            log_metrics(run, {'val/nll': validation()}, step)
        checkpoint(output / 'latest.pt')
    finally:
        run.finish()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='dap-challenge/configs/bc.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', required=True, choices=('cpu', 'mps', 'cuda'))
    parser.add_argument('--wandb-project')
    parser.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'))
    parser.add_argument('--resume')
    args = parser.parse_args()
    train(load_config(args.config), args.output, args.device, args.wandb_project, args.wandb_mode, args.resume)


if __name__ == '__main__':
    main()
