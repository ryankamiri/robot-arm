"""Observation fusion and Gaussian heads for the dap actor."""

from jaxtyping import Float
from torch import Tensor, nn

from models.architecture import GaussianParameters, ObservationEncoder
from models.dimensions import A, D_FUSION, D_OBS


class DapActorNetwork(ObservationEncoder):
    """Fuse encoded observation history and predict pre-tanh Gaussian parameters."""

    def __init__(self, image_encoder: nn.Module, history_length: int) -> None:
        super().__init__(image_encoder, history_length)
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
        images: Float[Tensor, "B L I C_in H_in W_in"],
        positions: Float[Tensor, "B L A"],
    ) -> GaussianParameters:
        """Preserve actor parameter names while reusing observation encoding."""
        flattened: Float[Tensor, "B D_flat"] = super().forward(images, positions)
        features: Float[Tensor, "B D_FUSION"] = self.fusion(flattened)
        return GaussianParameters(
            mean=self.mean_head(features),
            log_std=self.log_std_head(features),
        )
