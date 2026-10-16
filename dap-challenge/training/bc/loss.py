"""Expert-action likelihood for the single-action tanh-Gaussian actor."""

import math

from jaxtyping import Float
import torch
from torch import Tensor
from torch.distributions import Normal

from models.architecture import GaussianParameters
from models.distributions import tanh_gaussian_log_prob


def expert_nll(
    parameters: GaussianParameters, expert_actions: Float[Tensor, 'B A'], epsilon: float,
) -> Float[Tensor, 'B']:
    """Sum channel NLLs per sample; clamp labels only for likelihood evaluation."""
    if not math.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError('epsilon must be finite and strictly between zero and one')
    if expert_actions.ndim != 2 or expert_actions.shape != parameters.mean.shape:
        raise ValueError('expert actions and Gaussian parameters must have shape [B,A]')
    if parameters.log_std.shape != parameters.mean.shape:
        raise ValueError('mean and log_std shapes must match')
    if not torch.isfinite(expert_actions).all() or (expert_actions.abs() > 1).any():
        raise ValueError('expert actions must be finite and lie in [-1,1]')

    targets: Float[Tensor, 'B A'] = expert_actions.clamp(-1 + epsilon, 1 - epsilon)
    if (targets.abs() >= 1).any():
        raise ValueError('epsilon is too small for the target dtype')

    pre_tanh: Float[Tensor, 'B A'] = targets.atanh()
    distribution = Normal(parameters.mean, parameters.log_std.exp())
    return -tanh_gaussian_log_prob(distribution, pre_tanh)
