"""Expert-action likelihood for the single-action tanh-Gaussian actor."""

import math

from jaxtyping import Bool, Float
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


def weighted_nll(
    nll: Float[Tensor, 'B'], non_waiting: Bool[Tensor, 'B'], waiting_weight: float, non_waiting_weight: float,
) -> Float[Tensor, '']:
    """Reweight per-observation NLL without changing targets or dropping examples."""
    if nll.ndim != 1 or nll.numel() == 0 or non_waiting.shape != nll.shape or non_waiting.dtype != torch.bool:
        raise ValueError('NLL and boolean activity labels must have matching nonempty [B] shapes')
    for weight in (waiting_weight,non_waiting_weight):
        if isinstance(weight,bool) or not isinstance(weight,(int,float)) or not math.isfinite(weight) or weight<=0:
            raise ValueError('activity loss weights must be positive and finite')

    # w_i = w_non_waiting if observation i is active, otherwise w_waiting.
    weights: Float[Tensor, 'B'] = torch.where(non_waiting.to(nll.device),
                                             nll.new_tensor(non_waiting_weight),nll.new_tensor(waiting_weight))
    # L = sum_i(w_i * ell_i) / sum_i(w_i), where ell_i = -log pi_theta(a_i_expert | o_i).
    return (weights*nll).sum()/weights.sum()
