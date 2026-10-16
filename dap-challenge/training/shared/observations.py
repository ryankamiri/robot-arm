"""RGB preprocessing and causal histories shared by BC and future RL."""

from dataclasses import dataclass

from jaxtyping import Float, Int64, UInt8
import torch
from torch import Tensor
from torch.nn import functional as F

from models.dimensions import A, C_RGB, I_INPUTS


@dataclass
class ObservationHistory:
    """One sample's L-observation window, oldest to newest; not the full episode."""

    images: Float[Tensor, "L I C_in H_in W_in"]
    positions: Float[Tensor, "L A"]


def preprocess_image(
    image: UInt8[Tensor, "H_src W_src C_in"],
    image_width: int,
    image_height: int,
    rotation_clockwise: int = 0,
) -> Float[Tensor, "C_in H_in W_in"]:
    """Undo the recorded clockwise rotation, then resize RGB to CHW float32 [0,1]."""
    if any(isinstance(size, bool) or not isinstance(size, int) or size < 1
           for size in (image_width, image_height)):
        raise ValueError("image dimensions must be positive integers")
    if image.ndim != 3 or image.shape[-1] != C_RGB or min(image.shape) < 1:
        raise ValueError(f"image must be RGB [H_src,W_src,C_in], C_in={C_RGB}")
    if image.dtype != torch.uint8:
        raise TypeError("preprocess_image expects uint8 RGB pixels")

    if isinstance(rotation_clockwise, bool) or rotation_clockwise not in (0, 90, 180, 270):
        raise ValueError('recorded rotation must be 0, 90, 180 or 270 degrees')

    image = torch.rot90(image, k=rotation_clockwise // 90, dims=(0, 1))
    pixels: Float[Tensor, "B C_in H_src W_src"] = (
        image.permute(2, 0, 1).to(torch.float32).unsqueeze(0) / 255.0
    )
    resized: Float[Tensor, "B C_in H_in W_in"] = F.interpolate(
        pixels,
        size=(image_height, image_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return resized.squeeze(0)


def build_history(
    frames: tuple[Float[Tensor, "F_arm C_in H_in W_in"], Float[Tensor, "F_mac C_in H_in W_in"]],
    frame_indices: Int64[Tensor, "T I"],
    positions: Float[Tensor, "T A"],
    time_index: int,
    history_length: int,
) -> ObservationHistory:
    """Select both views per recorded row, repeating the first complete observation."""
    if isinstance(history_length, bool) or not isinstance(history_length, int) or history_length < 1:
        raise ValueError("history_length must be a positive integer")
    arm_frames, mac_frames = frames
    if any(view.ndim != 4 or view.shape[1] != C_RGB for view in (arm_frames, mac_frames)):
        raise ValueError("frames must contain two decoded camera tensors [F,C_in,H_in,W_in]")
    if positions.ndim != 2 or positions.shape[1] != A:
        raise ValueError(f"positions must have shape [T,A], A={A}")

    T = positions.shape[0]
    if frame_indices.shape != (T, I_INPUTS) or frame_indices.dtype != torch.int64:
        raise ValueError("frame_indices must be int64 [T,I], I=2")
    if isinstance(time_index, bool) or not isinstance(time_index, int) or not 0 <= time_index < T:
        raise IndexError("time_index is outside this episode")
    if frame_indices.device != positions.device or any(view.device != positions.device for view in frames):
        raise ValueError("history tensors must share a device")
    if frames[0].shape[1:] != frames[1].shape[1:]:
        raise ValueError("camera frames must have matching preprocessed image shapes")

    rows: Int64[Tensor, "L"] = torch.arange(
        time_index - history_length + 1, time_index + 1, device=positions.device
    ).clamp_min(0)
    selected: Int64[Tensor, "L I"] = frame_indices.index_select(0, rows)
    views = []
    for i, view in ((0, arm_frames), (1, mac_frames)):
        indices = selected[:, i]
        if (indices < 0).any() or (indices >= view.shape[0]).any():
            raise ValueError("history references a nonexistent video frame")

        views.append(view.index_select(0, indices))
    return ObservationHistory(images=torch.stack(views, dim=1), positions=positions.index_select(0, rows))
