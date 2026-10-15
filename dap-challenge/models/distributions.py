"""Differentiable tanh-Gaussian sampling in normalized action coordinates.

B is batch size and A is action channels; see dimensions.py for fixed values.
"""

from dataclasses import dataclass
import math

from jaxtyping import Float
from torch import Tensor
from torch.distributions import Normal
from torch.nn import functional as F

from .architecture import GaussianParameters


@dataclass
class ActionSample:
    """A single action per example, with its density under the actor.

    actions and pre_tanh have shape [B, A]. log_prob has shape [B] after
    summing log densities over the six independent base-Gaussian channels.
    """

    actions: Float[Tensor, "B A"]
    log_prob: Float[Tensor, "B"]
    pre_tanh: Float[Tensor, "B A"]


def sample_tanh_gaussian(parameters: GaussianParameters) -> ActionSample:
    """Reparameterized sampling; gradients reach both Gaussian parameter heads.

    Compute the change-of-variables correction from the pre-tanh sample instead
    of inverting the bounded action. This remains stable when floating-point
    tanh rounds a large latent sample to exactly -1 or 1. No action clipping or
    arbitrary epsilon is needed for log probabilities of our own samples.
    """
    distribution = Normal(parameters.mean, parameters.log_std.exp())
    pre_tanh: Float[Tensor, "B A"] = distribution.rsample()
    actions: Float[Tensor, "B A"] = pre_tanh.tanh()

    # log(1 - tanh(u)^2), evaluated without subtracting nearly equal values.
    log_jacobian: Float[Tensor, "B A"] = 2 * (
        math.log(2.0) - pre_tanh - F.softplus(-2 * pre_tanh)
    )
    log_prob: Float[Tensor, "B"] = (
        distribution.log_prob(pre_tanh) - log_jacobian
    ).sum(dim=-1)
    return ActionSample(actions=actions, log_prob=log_prob, pre_tanh=pre_tanh)
