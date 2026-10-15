"""Observation fusion and Gaussian heads for the dap actor.

The image encoder is supplied separately. This keeps the residual CNN distinct
from the history/position fusion and from the action-distribution interface.
See dimensions.py for the definitions of all symbolic tensor axes.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor, nn

from .dimensions import A, C_RGB, D_FUSION, D_IMG, D_OBS, D_POS, J


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions, a matching shortcut, and ReLU; no normalization."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        # Every stage changes shape, so project the shortcut to match.
        self.shortcut = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=stride
        )
        self.activation = nn.ReLU()

    def forward(
        self, images: Float[Tensor, "B C_in H_in W_in"]
    ) -> Float[Tensor, "B C_out H_out W_out"]:
        residual: Float[Tensor, "B C_out H_out W_out"] = (
            self.shortcut(images)
        )
        features: Float[Tensor, "B C_out H_out W_out"] = (
            self.activation(self.conv1(images))
        )
        return self.activation(self.conv2(features) + residual)


class ImageEncoder(nn.Module):
    """RGB image -> residual stages -> global pooling -> 128 features.

    Every stage downsamples by two. Adaptive pooling avoids selecting a fixed
    camera resolution here; training and deployment still need matching image
    preprocessing. Parameters use the standard PyTorch layer initialization.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stages = nn.Sequential(
            ResidualBlock(C_RGB, 32, stride=2),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, D_IMG, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(
        self, images: Float[Tensor, "B C_in H_in W_in"]
    ) -> Float[Tensor, "B D_IMG"]:
        B = images.shape[0]
        return self.pool(self.stages(images)).reshape(B, D_IMG)


@dataclass
class GaussianParameters:
    """Parameters of the pre-tanh Gaussian, each shaped [B, A]."""

    mean: Float[Tensor, "B A"]
    log_std: Float[Tensor, "B A"]


class DapActorNetwork(nn.Module):
    """Encode each observation, flatten history, and predict Gaussian parameters.

    Args:
        image_encoder: Shared encoder mapping [N, C_in, H_in, W_in] to [N, D_IMG].
        history_length: Total observations per window, including the current one.
            Required explicitly because the project's history length is open.

    Images must already be floating-point tensors in the caller's chosen image
    preprocessing convention. Positions must already use the selected [-1, 1]
    representation. This network does not read sensors or command the robot.
    """

    def __init__(self, image_encoder: nn.Module, history_length: int) -> None:
        super().__init__()
        if isinstance(history_length, bool) or not isinstance(history_length, int):
            raise TypeError("history_length must be an integer")
        if history_length < 1:
            raise ValueError("history_length must be positive")

        self.history_length = history_length
        self.image_encoder = image_encoder
        self.position_encoder = nn.Sequential(
            nn.Linear(J, D_POS),
            nn.ReLU(),
            nn.Linear(D_POS, D_POS),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(history_length * D_OBS, 256),
            nn.ReLU(),
            nn.Linear(256, D_FUSION),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(D_FUSION, A)
        self.log_std_head = nn.Linear(D_FUSION, A)

    def forward(
        self,
        images: Float[Tensor, "B L C_in H_in W_in"],
        positions: Float[Tensor, "B L J"],
    ) -> GaussianParameters:
        """Map images [B,L,C_in,H_in,W_in] and positions [B,L,J] to two [B,A] tensors."""
        if images.ndim != 5:
            raise ValueError("images must have shape [B, L, C_in, H_in, W_in]")
        B, L, C_in, H_in, W_in = images.shape
        if B < 1 or H_in < 1 or W_in < 1:
            raise ValueError("image batch and spatial dimensions must be positive")
        if L != self.history_length or C_in != C_RGB:
            raise ValueError(
                f"expected L={self.history_length}, C_in={C_RGB}; "
                f"got L={L}, C_in={C_in}"
            )
        if positions.shape != (B, L, J):
            raise ValueError(f"positions must match images with shape [B, L, J], J={J}")
        if not images.is_floating_point() or not positions.is_floating_point():
            raise TypeError("images and positions must be floating-point tensors")
        if images.device != positions.device or images.dtype != positions.dtype:
            raise ValueError("images and positions must share a device and dtype")

        # Folding time into the batch applies the SAME encoder to every frame.
        N = B * L
        encoded_images: Float[Tensor, "N D_IMG"] = self.image_encoder(
            images.reshape(N, C_in, H_in, W_in)
        )
        if encoded_images.shape != (N, D_IMG):
            raise ValueError(
                f"image_encoder must return [N, D_IMG], N=B*L and D_IMG={D_IMG}"
            )
        image_features: Float[Tensor, "B L D_IMG"] = encoded_images.reshape(
            B, L, D_IMG
        )
        position_features: Float[Tensor, "B L D_POS"] = self.position_encoder(
            positions
        )
        per_observation: Float[Tensor, "B L D_OBS"] = torch.cat(
            (image_features, position_features), dim=-1
        )
        D_flat = L * D_OBS
        flattened: Float[Tensor, "B D_flat"] = per_observation.reshape(B, D_flat)
        features: Float[Tensor, "B D_FUSION"] = self.fusion(flattened)
        return GaussianParameters(
            mean=self.mean_head(features),
            log_std=self.log_std_head(features),
        )
