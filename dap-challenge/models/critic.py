"""Twin action-value critics with independent BC-initialized observation encoders."""

from dataclasses import dataclass
from typing import Self

from jaxtyping import Float
import torch
from torch import Tensor, nn

from models.architecture import ImageEncoder, ObservationEncoder
from models.dimensions import A, D_FUSION, D_OBS
from models.policy import DapPolicy


class DapCritic(ObservationEncoder):
    """Encode history plus a normalized action and predict one unbounded soft Q-value."""

    def __init__(self, history_length: int) -> None:
        super().__init__(ImageEncoder(), history_length)
        self.fusion = nn.Sequential(
            nn.Linear(history_length * D_OBS + A, 256),
            nn.ReLU(),
            nn.Linear(256, D_FUSION),
            nn.ReLU(),
        )
        self.q_head = nn.Linear(D_FUSION, 1)

    def initialize_from_policy(self, policy: DapPolicy) -> None:
        """Copy BC image/position weights without aliasing parameters or copying Gaussian heads."""
        if self.history_length != policy.history_length:
            raise ValueError('critic and BC policy must have matching history lengths')

        source = next(policy.parameters())
        self.to(device=source.device, dtype=source.dtype)
        self.image_encoder.load_state_dict(policy.network.image_encoder.state_dict())
        self.position_encoder.load_state_dict(policy.network.position_encoder.state_dict())

    @classmethod
    def from_policy(cls, policy: DapPolicy) -> Self:
        """Start with trained observation encoders and a fresh action-conditioned value network."""
        critic = cls(policy.history_length)
        critic.initialize_from_policy(policy)
        return critic

    def forward(
        self,
        images: Float[Tensor, 'B L I C_in H_in W_in'],
        positions: Float[Tensor, 'B L A'],
        actions: Float[Tensor, 'B A'],
    ) -> Float[Tensor, 'B']:
        """Concatenate observation features [B,D_flat] and candidate actions [B,A]."""
        observations: Float[Tensor, 'B D_flat'] = super().forward(images, positions)
        B = observations.shape[0]
        if actions.shape != (B, A):
            raise ValueError('critic actions must have shape [B,A] matching observations')
        if not actions.is_floating_point():
            raise TypeError('critic actions must be floating-point tensors')
        if actions.device != images.device or actions.dtype != images.dtype:
            raise ValueError('critic actions and observations must share a device and dtype')
        if not torch.isfinite(actions).all() or (actions.abs() > 1).any():
            raise ValueError('critic actions must be finite normalized targets in [-1,1]')

        # Q_phi(x,a): D_input=D_flat+A; [B,D_input] -> [B,D_FUSION] -> [B].
        inputs: Float[Tensor, 'B D_input'] = torch.cat((observations, actions), dim=-1)
        features: Float[Tensor, 'B D_FUSION'] = self.fusion(inputs)
        return self.q_head(features).reshape(B)


@dataclass
class TwinQValues:
    """Two estimates of the same soft action value; each output has shape [B]."""

    q1: Float[Tensor, 'B']
    q2: Float[Tensor, 'B']


class TwinCritics(nn.Module):
    """Separate critics with matching architectures and independently initialized value layers."""

    def __init__(self, history_length: int) -> None:
        super().__init__()
        self.q1 = DapCritic(history_length)
        self.q2 = DapCritic(history_length)

    @classmethod
    def from_policy(cls, policy: DapPolicy) -> Self:
        """Initialize both observation encoders from BC, keeping all parameter storage separate."""
        critics = cls(policy.history_length)
        critics.q1.initialize_from_policy(policy)
        critics.q2.initialize_from_policy(policy)
        return critics

    def forward(
        self,
        images: Float[Tensor, 'B L I C_in H_in W_in'],
        positions: Float[Tensor, 'B L A'],
        actions: Float[Tensor, 'B A'],
    ) -> TwinQValues:
        """Return Q1(x,a) and Q2(x,a) without reducing the two estimates."""
        return TwinQValues(self.q1(images, positions, actions), self.q2(images, positions, actions))
