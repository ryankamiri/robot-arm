"""Offline activity labels and sample-weighted action metrics shared by training loops."""

from collections.abc import Sequence
import math

from jaxtyping import Bool, Float
import torch
from torch import Tensor

from models.dimensions import A
from shared.normalization import controller_bounds


def non_waiting_mask(
    positions: Float[Tensor, 'T A'], targets: Float[Tensor, 'T A'],
    timestamps: Float[Tensor, 'T'], joint_names: Sequence[str],
    motion_threshold: float, window_half_s: float,
) -> Bool[Tensor, 'T']:
    """Label motion or target changes over a configured centered window, after 3-row median smoothing."""
    for name,value in (('motion_threshold',motion_threshold),('window_half_s',window_half_s)):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError(f'{name} must be positive and finite')

    if positions.ndim != 2 or positions.shape[1] != A or targets.shape != positions.shape:
        raise ValueError('positions and targets must have matching [T,A] shapes')
    if len(positions) == 0 or timestamps.shape != (len(positions),):
        raise ValueError('activity labels require nonempty matching timestamps [T]')
    if not all(torch.isfinite(values).all() for values in (positions, targets, timestamps)):
        raise ValueError('activity inputs must be finite')
    if (torch.diff(timestamps) <= 0).any():
        raise ValueError('activity timestamps must strictly increase')

    lower, upper = controller_bounds(joint_names)
    scale: Float[Tensor, 'A'] = (torch.tensor(upper, dtype=torch.float64)-torch.tensor(lower, dtype=torch.float64))/2
    active: Bool[Tensor, 'T'] = torch.zeros(len(timestamps), dtype=torch.bool)
    for values in (positions, targets):
        normalized: Float[Tensor, 'T A'] = values.to(dtype=torch.float64)/scale
        smooth: Float[Tensor, 'T A'] = torch.stack([
            torch.quantile(normalized[max(0,index-1):min(len(values),index+2)],0.5,dim=0)
            for index in range(len(values))
        ])
        for index,timestamp in enumerate(timestamps):
            window = smooth[(timestamps-timestamp).abs()<=window_half_s]
            active[index] |= (window.amax(dim=0)-window.amin(dim=0)).max()>motion_threshold
    return active


def activity_mae_totals(
    errors: Float[Tensor, 'B'], non_waiting: Bool[Tensor, 'B'],
) -> dict[str, tuple[float, int]]:
    """Return sums and counts so partial batches and absent classes aggregate correctly."""
    if errors.ndim != 1 or non_waiting.shape != errors.shape or non_waiting.dtype != torch.bool:
        raise ValueError('action MAE and boolean activity labels must have matching [B] shapes')

    non_waiting = non_waiting.to(errors.device)
    return {name:(errors[mask].sum().item(),int(mask.sum().item()))
            for name,mask in (('waiting',~non_waiting),('non_waiting',non_waiting))}


def activity_mae_metrics(totals: dict[str, tuple[float, int]], prefix: str) -> dict[str, float | int]:
    """Omit absent-group MAE instead of logging zero or NaN; always expose group counts."""
    metrics = {}
    for name,(error_sum,count) in totals.items():
        if count < 0 or not math.isfinite(error_sum):
            raise ValueError('activity metric totals must be finite with nonnegative counts')

        metrics[f'{prefix}/{name}_count'] = count
        if count:
            metrics[f'{prefix}/action_mae_{name}'] = error_sum/count
    return metrics
