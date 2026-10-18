"""Convert explicit LeRobot calibrated range values at the model boundary."""

from collections.abc import Sequence

from jaxtyping import Float
import torch
from torch import Tensor

from models.dimensions import A

JOINT_NAMES = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper",
)


def controller_bounds(joint_names: Sequence[str]) -> tuple[list[float], list[float]]:
    """Return calibrated controller bounds in the episode's explicit joint order."""
    if len(joint_names) != A or set(joint_names) != set(JOINT_NAMES):
        raise ValueError("joint_names must contain each SO-101 joint exactly once")

    return ([-100.0 if name != "gripper" else 0.0 for name in joint_names], [100.0] * A)


def normalize_controller(
    values: Float[Tensor, "... A"], joint_names: Sequence[str]
) -> Float[Tensor, "... A"]:
    """Normalize calibrated values without silently clipping invalid measurements."""
    if values.ndim < 1 or values.shape[-1] != A or not values.is_floating_point():
        raise ValueError("controller values must be floating-point [...,A]")
    if not torch.isfinite(values).all():
        raise ValueError("controller values must be finite")

    lower, upper = controller_bounds(joint_names)
    minimum: Float[Tensor, "A"] = values.new_tensor(lower)
    maximum: Float[Tensor, "A"] = values.new_tensor(upper)
    if ((values < minimum) | (values > maximum)).any():
        raise ValueError("controller values lie outside the declared calibrated range")

    return 2 * (values - minimum) / (maximum - minimum) - 1


def denormalize_controller(
    values: Float[Tensor, "... A"], joint_names: Sequence[str]
) -> Float[Tensor, "... A"]:
    """Invert model normalization using the same explicit controller convention."""
    if values.ndim < 1 or values.shape[-1] != A or not values.is_floating_point():
        raise ValueError("normalized values must be floating-point [...,A]")
    if not torch.isfinite(values).all() or (values.abs() > 1).any():
        raise ValueError("normalized values must be finite and in [-1,1]")

    lower, upper = controller_bounds(joint_names)
    minimum: Float[Tensor, "A"] = values.new_tensor(lower)
    maximum: Float[Tensor, "A"] = values.new_tensor(upper)
    return minimum + (values + 1) * (maximum - minimum) / 2
