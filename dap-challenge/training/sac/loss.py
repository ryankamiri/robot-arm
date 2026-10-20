"""SAC objectives in requested normalized action coordinates."""

from jaxtyping import Bool, Float
import torch
from torch import Tensor

from models.critic import TwinQValues


def critic_target(
    rewards: Float[Tensor, 'B'], terminated: Bool[Tensor, 'B'],
    next_values: TwinQValues, next_log_prob: Float[Tensor, 'B'],
    alpha: Tensor, discount: float,
) -> Float[Tensor, 'B']:
    """Compute a detached one-step soft Bellman target, with no terminal bootstrap."""
    # y = r + gamma * (1 - terminated) * (min(Q1_target,Q2_target) - alpha * log pi).
    continuation = torch.min(next_values.q1, next_values.q2) - alpha * next_log_prob
    return (rewards + discount * torch.where(terminated, torch.zeros_like(continuation), continuation)).detach()


def critic_loss(values: TwinQValues, targets: Float[Tensor, 'B']) -> Tensor:
    """Sum the two mean squared Bellman regression losses."""
    # L_Q = mean((Q1-y)^2) + mean((Q2-y)^2).
    return (values.q1 - targets).square().mean() + (values.q2 - targets).square().mean()


def actor_loss(values: TwinQValues, log_prob: Float[Tensor, 'B'], alpha: Tensor) -> Tensor:
    """Keep action gradients through Q while updating only the actor parameters."""
    # L_pi = mean(alpha * log pi(a|x) - min(Q1(x,a),Q2(x,a))).
    return (alpha.detach() * log_prob - torch.min(values.q1, values.q2)).mean()


def temperature_loss(log_alpha: Tensor, log_prob: Float[Tensor, 'B'], target_entropy: float) -> Tensor:
    """Tune log temperature against detached sampled entropy estimates."""
    # L_log_alpha = -mean(log_alpha * stop_gradient(log pi + H_target)).
    return -(log_alpha * (log_prob.detach() + target_entropy)).mean()
