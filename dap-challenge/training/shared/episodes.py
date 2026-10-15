"""Local episode storage read through Hugging Face Datasets and PyAV."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator

import av
from datasets import Dataset
from jaxtyping import Float, Int64, UInt8
import pyarrow.parquet as pq
import torch
from torch import Tensor

from models.dimensions import A
from .observations import preprocess_image


@dataclass
class Episode:
    """One episode's table, normalized robot values and resized video frames."""

    episode_id: str
    metadata: dict[str, Any]
    table: Dataset
    frames: Float[Tensor, "V C_in H_in W_in"]
    frame_indices: Int64[Tensor, "T"]
    positions: Float[Tensor, "T A"]
    actions: Float[Tensor, "T A"]

    def __len__(self) -> int:
        """Return observation/action row count, not video frame count."""
        return len(self.table)


def discover_episodes(root: str | Path) -> list[Path]:
    """Find sorted child episode directories marked by metadata.json."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)

    paths = sorted(path.parent for path in root.glob("*/metadata.json"))
    if not paths:
        raise ValueError(f"no episode directories found under {root}")

    return paths


def _video_frames(path: Path) -> Iterator[UInt8[Tensor, "H_src W_src C_in"]]:
    """Decode display-order RGB frames without inferring alignment from FPS."""
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"no video stream in {path}")

        for frame in container.decode(video=0):
            yield torch.from_numpy(frame.to_ndarray(format="rgb24"))


def load_episode(
    directory: str | Path, image_width: int, image_height: int
) -> Episode:
    """Validate an episode and load its already-normalized observations/targets."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("metadata must declare schema_version=1")

    episode_id = metadata.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("metadata must contain a nonempty episode_id")

    joint_names = metadata.get("joint_names")
    if (not isinstance(joint_names, list) or len(joint_names) != A
            or not all(isinstance(name, str) and name for name in joint_names)
            or len(set(joint_names)) != A):
        raise ValueError(f"joint_names must list A={A} unique names in model order")
    if metadata.get("normalized") is not True:
        raise ValueError("metadata must declare normalized=true for positions and actions")

    # Construct directly from the local Arrow table: no Hub request/cache build.
    table = Dataset(pq.read_table(directory / "steps.parquet"))
    required = {
        "time_index", "frame_index", "timestamp_s", "camera_timestamp_s",
        "action_timestamp_s", "positions", "actions",
    }
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(f"missing Parquet columns: {sorted(missing)}")

    T = len(table)
    if T == 0:
        raise ValueError("episodes must contain at least one observation/action row")

    columns = table.to_dict()
    for name in ("time_index", "frame_index"):
        if any(isinstance(value, bool) or not isinstance(value, int) for value in columns[name]):
            raise ValueError(f"{name} must contain integer indices")
    if columns["time_index"] != list(range(T)):
        raise ValueError("time_index must be consecutive from zero")

    frame_indices: Int64[Tensor, "T"] = torch.tensor(columns["frame_index"], dtype=torch.int64)
    if (frame_indices < 0).any() or (frame_indices.diff() < 0).any():
        raise ValueError("frame_index must be nonnegative and nondecreasing")

    timestamp: Float[Tensor, "T"] = torch.tensor(columns["timestamp_s"], dtype=torch.float64)
    camera_timestamp: Float[Tensor, "T"] = torch.tensor(columns["camera_timestamp_s"], dtype=torch.float64)
    action_timestamp: Float[Tensor, "T"] = torch.tensor(columns["action_timestamp_s"], dtype=torch.float64)
    for name, values in (
        ("timestamp_s", timestamp),
        ("camera_timestamp_s", camera_timestamp),
        ("action_timestamp_s", action_timestamp),
    ):
        if values.shape != (T,) or not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name} must contain finite nonnegative seconds [T]")
        if (values.diff() < 0).any():
            raise ValueError(f"{name} must be nondecreasing")
    if (timestamp.diff() <= 0).any():
        raise ValueError("observation timestamps must be strictly increasing")
    if (camera_timestamp > timestamp).any():
        raise ValueError("camera capture cannot be later than its observation")
    if (action_timestamp < timestamp).any():
        raise ValueError("action send time cannot be earlier than its observation")

    captures: dict[int, float] = {}
    for index, capture_time in zip(columns["frame_index"], columns["camera_timestamp_s"]):
        if index in captures and captures[index] != capture_time:
            raise ValueError("a reused video frame must retain its capture timestamp")

        captures[index] = capture_time

    positions: Float[Tensor, "T A"] = torch.tensor(columns["positions"], dtype=torch.float32)
    actions: Float[Tensor, "T A"] = torch.tensor(columns["actions"], dtype=torch.float32)
    for name, values in (("positions", positions), ("actions", actions)):
        if values.shape != (T, A) or not torch.isfinite(values).all() or (values.abs() > 1).any():
            raise ValueError(f"{name} must be finite normalized values [T,A], A={A}, in [-1,1]")

    decoded: list[Float[Tensor, "C_in H_in W_in"]] = [
        preprocess_image(frame, image_width, image_height)
        for frame in _video_frames(directory / "video.mp4")
    ]
    if not decoded:
        raise ValueError("episode video contains no frames")

    frames: Float[Tensor, "V C_in H_in W_in"] = torch.stack(decoded)
    if frame_indices.max().item() >= frames.shape[0]:
        raise ValueError("frame_index references a nonexistent video frame")

    return Episode(episode_id, metadata, table, frames, frame_indices, positions, actions)
