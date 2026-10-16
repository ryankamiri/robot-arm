"""Differentiable tanh-Gaussian sampling in normalized action coordinates."""

from dataclasses import dataclass
import math

from jaxtyping import Float
from torch import Tensor
from torch.distributions import Normal
from torch.nn import functional as F

from .architecture import GaussianParameters


@dataclass
class ActionSample:
    """A single action per example, with its density under the actor."""

    actions: Float[Tensor, "B A"]
    log_prob: Float[Tensor, "B"]
    pre_tanh: Float[Tensor, "B A"]


def tanh_gaussian_log_prob(
    distribution: Normal, pre_tanh: Float[Tensor, "B A"],
) -> Float[Tensor, "B"]:
    """Compute the density of tanh(pre_tanh), summing action channels [B,A] to [B]."""
    # log(1 - tanh(u)^2), evaluated without subtracting nearly equal values.
    log_jacobian: Float[Tensor, "B A"] = 2 * (
        math.log(2.0) - pre_tanh - F.softplus(-2 * pre_tanh)
    )
    return (distribution.log_prob(pre_tanh) - log_jacobian).sum(dim=-1)


def sample_tanh_gaussian(parameters: GaussianParameters) -> ActionSample:
    """Reparameterized sampling; gradients reach both Gaussian parameter heads."""
    distribution = Normal(parameters.mean, parameters.log_std.exp())
    pre_tanh: Float[Tensor, "B A"] = distribution.rsample()
    actions: Float[Tensor, "B A"] = pre_tanh.tanh()

    log_prob: Float[Tensor, "B"] = tanh_gaussian_log_prob(distribution, pre_tanh)
    return ActionSample(actions=actions, log_prob=log_prob, pre_tanh=pre_tanh)
