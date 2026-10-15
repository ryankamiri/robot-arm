"""W&B run setup and scalar logging, shared by future BC and SAC loops.

Importing this module does not initialize W&B. A caller must explicitly choose a
project, mode and local output directory. Tests use disabled/offline modes only.
"""

from pathlib import Path
from typing import Any, Literal, Mapping

import wandb

from .config import _plain_config


def start_run(
    project: str,
    config: Mapping[str, Any],
    mode: Literal["online", "offline", "disabled"],
    directory: str | Path,
    entity: str | None = None,
    name: str | None = None,
    run_id: str | None = None,
    resume: Literal["allow", "must", "never"] | None = None,
) -> wandb.Run:
    """Start a caller-configured W&B run; the caller owns run.finish().

    Store run.id in checkpoint extra_state to associate a resumed trainer with
    its W&B run. Offline W&B resume is not equivalent to server-side run resume.
    """
    if not project.strip():
        raise ValueError("project must be nonempty")
    if mode not in ("online", "offline", "disabled"):
        raise ValueError("mode must be online, offline or disabled")
    if resume is not None and run_id is None:
        raise ValueError("an explicit run_id is required when selecting resume behavior")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=project,
        config=_plain_config(config),
        mode=mode,
        dir=str(directory),
        entity=entity,
        name=name,
        id=run_id,
        resume=resume,
    )
    run.define_metric("optimizer_step")
    run.define_metric("*", step_metric="optimizer_step")
    return run


def log_metrics(
    run: wandb.Run, metrics: Mapping[str, int | float], step: int
) -> None:
    """Log scalar values against optimizer updates, including train/val together.

    W&B's internal record index is separate from optimizer_step. Separate train
    and validation calls can therefore share an optimizer step without dropping
    a record. Convert scalar tensors using .item() before calling this helper.
    """
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative integer optimizer-update count")
    if "optimizer_step" in metrics:
        raise ValueError("optimizer_step is supplied by the step argument")
    if any(not isinstance(value, (int, float)) for value in metrics.values()):
        raise TypeError("metrics must contain numeric scalars; convert tensors with .item()")
    run.log({"optimizer_step": step, **metrics})
