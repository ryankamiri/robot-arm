"""Hugging Face-compatible actor interface, separate from robot execution."""

import math

from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Float
from torch import Tensor, nn

from .architecture import DapActorNetwork, GaussianParameters, ImageEncoder
from .distributions import ActionSample, sample_tanh_gaussian


class DapPolicy(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/ryankamiri/robot-arm",
    tags=["robotics", "so101", "reinforcement-learning"],
):
    """Single-action actor with forward/sample and Hub serialization methods."""

    def __init__(
        self,
        history_length: int,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        if not (
            math.isfinite(log_std_min)
            and math.isfinite(log_std_max)
            and log_std_min < log_std_max
        ):
            raise ValueError("log_std bounds must be finite and strictly ordered")

        self.history_length = history_length
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.network = DapActorNetwork(ImageEncoder(), history_length=history_length)

    def forward(
        self,
        images: Float[Tensor, "B L C_in H_in W_in"],
        positions: Float[Tensor, "B L A"],
    ) -> GaussianParameters:
        """Return pre-tanh mean and bounded log_std, both [B,A], without sampling."""
        parameters = self.network(images, positions)
        return GaussianParameters(
            mean=parameters.mean,
            log_std=parameters.log_std.clamp(self.log_std_min, self.log_std_max),
        )

    def sample(
        self,
        images: Float[Tensor, "B L C_in H_in W_in"],
        positions: Float[Tensor, "B L A"],
    ) -> ActionSample:
        """Return sampled normalized actions [B,A] and joint log_prob [B]."""
        return sample_tanh_gaussian(self(images, positions))

    def deterministic_action(
        self,
        images: Float[Tensor, "B L C_in H_in W_in"],
        positions: Float[Tensor, "B L A"],
    ) -> Float[Tensor, "B A"]:
        """Return tanh(mean) [B,A], not the exact mean of the transformed law."""
        return self(images, positions).mean.tanh()
