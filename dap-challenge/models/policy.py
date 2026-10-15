"""Hugging Face-compatible actor interface, separate from robot execution.

Shape symbols are defined in dimensions.py: B=batch, L=history, C_in=RGB
channels, H_in/W_in=image size, J=position channels, A=action channels.
"""

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
    """Single-action actor with forward/sample and Hub serialization methods.

    history_length counts all observations including the current observation.
    There is intentionally no default: constructing a model requires an explicit
    value, and that value is saved in config.json along with the scale bounds.

    Inputs are already-preprocessed floating-point RGB images [B,L,C_in,H_in,W_in] and
    normalized joint positions [B,L,J], in oldest-to-newest order. Returned
    actions are normalized absolute targets, not physical controller commands.
    """

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
        positions: Float[Tensor, "B L J"],
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
        positions: Float[Tensor, "B L J"],
    ) -> ActionSample:
        """Return sampled normalized actions [B,A] and joint log_prob [B].

        This method keeps gradients for SAC actor training. For robot inference,
        the caller should use eval() and torch.inference_mode(). It accepts the
        same observation tensors as forward(); no robot I/O happens here.
        """
        return sample_tanh_gaussian(self(images, positions))

    def deterministic_action(
        self,
        images: Float[Tensor, "B L C_in H_in W_in"],
        positions: Float[Tensor, "B L J"],
    ) -> Float[Tensor, "B A"]:
        """Return tanh(mean) [B,A], not the exact mean of the transformed law."""
        return self(images, positions).mean.tanh()
