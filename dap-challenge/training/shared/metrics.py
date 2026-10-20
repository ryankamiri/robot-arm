"""W&B run setup and scalar logging, shared by future BC and SAC loops."""

from pathlib import Path
from typing import Any, Literal, Mapping

import wandb

from shared.config import plain_config


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
    """Start a caller-configured W&B run; the caller owns run.finish()."""
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
        config=plain_config(config),
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
    """Log scalar values against optimizer updates, including train/val together."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative integer optimizer-update count")
    if "optimizer_step" in metrics:
        raise ValueError("optimizer_step is supplied by the step argument")
    if any(not isinstance(value, (int, float)) for value in metrics.values()):
        raise TypeError("metrics must contain numeric scalars; convert tensors with .item()")

    run.log({"optimizer_step": step, **metrics})
