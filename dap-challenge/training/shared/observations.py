"""RGB preprocessing and causal histories shared by BC and future RL."""

from dataclasses import dataclass

from jaxtyping import Float, Int64, UInt8
import torch
from torch import Tensor
from torch.nn import functional as F

from models.dimensions import A, C_RGB


@dataclass
class ObservationHistory:
    """One sample's L-observation window, oldest to newest; not the full episode."""

    images: Float[Tensor, "L C_in H_in W_in"]
    positions: Float[Tensor, "L A"]


def preprocess_image(
    image: UInt8[Tensor, "H_src W_src C_in"],
    image_width: int,
    image_height: int,
) -> Float[Tensor, "C_in H_in W_in"]:
    """Convert RGB bytes to CHW float32 [0,1], then bilinearly resize."""
    if any(isinstance(size, bool) or not isinstance(size, int) or size < 1
           for size in (image_width, image_height)):
        raise ValueError("image dimensions must be positive integers")
    if image.ndim != 3 or image.shape[-1] != C_RGB or min(image.shape) < 1:
        raise ValueError(f"image must be RGB [H_src,W_src,C_in], C_in={C_RGB}")
    if image.dtype != torch.uint8:
        raise TypeError("preprocess_image expects uint8 RGB pixels")

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
    frames: Float[Tensor, "V C_in H_in W_in"],
    frame_indices: Int64[Tensor, "T"],
    positions: Float[Tensor, "T A"],
    time_index: int,
    history_length: int,
) -> ObservationHistory:
    """Build consecutive history through time_index, repeating the first observation."""
    if isinstance(history_length, bool) or not isinstance(history_length, int) or history_length < 1:
        raise ValueError("history_length must be a positive integer")
    if frames.ndim != 4 or frames.shape[1] != C_RGB:
        raise ValueError("frames must have shape [V,C_in,H_in,W_in]")
    if positions.ndim != 2 or positions.shape[1] != A:
        raise ValueError(f"positions must have shape [T,A], A={A}")

    T = positions.shape[0]
    if frame_indices.shape != (T,) or frame_indices.dtype != torch.int64:
        raise ValueError("frame_indices must be int64 [T], matching positions")
    if isinstance(time_index, bool) or not isinstance(time_index, int) or not 0 <= time_index < T:
        raise IndexError("time_index is outside this episode")
    if not (frames.device == positions.device == frame_indices.device):
        raise ValueError("history tensors must share a device")

    rows: Int64[Tensor, "L"] = torch.arange(
        time_index - history_length + 1, time_index + 1, device=positions.device
    ).clamp_min(0)
    selected_frames: Int64[Tensor, "L"] = frame_indices.index_select(0, rows)
    if (selected_frames < 0).any() or (selected_frames >= frames.shape[0]).any():
        raise ValueError("history references a nonexistent video frame")

    return ObservationHistory(
        images=frames.index_select(0, selected_frames),
        positions=positions.index_select(0, rows),
    )
