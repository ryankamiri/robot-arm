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
from shared.observations import preprocess_image
from shared.normalization import normalize_controller


@dataclass
class Episode:
    """One episode's table, normalized robot values and resized video frames."""

    episode_id: str
    metadata: dict[str, Any]
    table: Dataset
    frames: tuple[Float[Tensor, "F_arm C_in H_in W_in"], Float[Tensor, "F_mac C_in H_in W_in"]]
    frame_indices: Int64[Tensor, "T I"]
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


def video_frames(path: Path) -> Iterator[UInt8[Tensor, "H_src W_src C_in"]]:
    """Decode display-order RGB frames without inferring alignment from FPS."""
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"no video stream in {path}")

        for frame in container.decode(video=0):
            yield torch.from_numpy(frame.to_ndarray(format="rgb24"))


def load_episode(
    directory: str | Path, image_width: int, image_height: int
) -> Episode:
    """Validate an episode and normalize calibrated observations and targets only during loading."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 2:
        raise ValueError("metadata must declare schema_version=2")

    episode_id = metadata.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("metadata must contain a nonempty episode_id")

    joint_names = metadata.get("joint_names")
    if (not isinstance(joint_names, list) or len(joint_names) != A
            or not all(isinstance(name, str) and name for name in joint_names)
            or len(set(joint_names)) != A):
        raise ValueError(f"joint_names must list A={A} unique names in model order")
    if metadata.get("normalized") is not False or metadata.get("controller_convention") != "lerobot_range":
        raise ValueError("metadata must declare normalized=false and controller_convention=lerobot_range")
    if not isinstance(metadata.get("calibration"), dict):
        raise ValueError("metadata must preserve calibration information")
    if metadata.get("status") != "accepted":
        raise ValueError("only accepted episodes can be loaded for training")

    # Construct directly from the local Arrow table: no Hub request/cache build.
    table = Dataset(pq.read_table(directory / "steps.parquet"))
    required = {
        "time_index", "frame_index", "timestamp_s", "camera_timestamp_s",
        "action_timestamp_s", "positions", "actions", "mac_frame_index", "mac_camera_timestamp_s",
    }
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(f"missing Parquet columns: {sorted(missing)}")

    T = len(table)
    if T == 0:
        raise ValueError("episodes must contain at least one observation/action row")

    columns = table.to_dict()
    for name in ("time_index", "frame_index", "mac_frame_index"):
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

    mac_indices: Int64[Tensor, "T"] = torch.tensor(columns['mac_frame_index'], dtype=torch.int64)
    mac_timestamp: Float[Tensor, "T"] = torch.tensor(columns['mac_camera_timestamp_s'], dtype=torch.float64)
    if (mac_indices < 0).any() or (mac_indices.diff() < 0).any():
        raise ValueError('mac_frame_index must be nonnegative and nondecreasing')
    if (not torch.isfinite(mac_timestamp).all() or (mac_timestamp < 0).any()
            or (mac_timestamp.diff() < 0).any() or (mac_timestamp > timestamp).any()):
        raise ValueError('Mac camera timestamps must be finite, nonnegative, ordered and no later than observations')

    captures = {}
    for index, capture_time in zip(columns['mac_frame_index'], columns['mac_camera_timestamp_s']):
        if index in captures and captures[index] != capture_time:
            raise ValueError('a reused Mac frame must retain its capture timestamp')

        captures[index] = capture_time

    positions: Float[Tensor, "T A"] = torch.tensor(columns["positions"], dtype=torch.float32)
    actions: Float[Tensor, "T A"] = torch.tensor(columns["actions"], dtype=torch.float32)
    for name, values in (("positions", positions), ("actions", actions)):
        if values.shape != (T, A) or not torch.isfinite(values).all():
            raise ValueError(f"{name} must be finite calibrated values [T,A], A={A}")

    positions = normalize_controller(positions, joint_names)
    actions = normalize_controller(actions, joint_names)

    views = []
    for name, filename, indices in (
        ('camera', 'video.mp4', frame_indices),
        ('mac_camera', 'mac_video.mp4', mac_indices),
    ):
        info = metadata.get(name)
        if not isinstance(info, dict):
            raise ValueError(f'{name} metadata is required for two-camera training')

        rotation = info.get('rotation_clockwise')
        if isinstance(rotation, bool) or not isinstance(rotation, int) or rotation not in (0, 90, 180, 270):
            raise ValueError(f'{name} must declare a valid rotation_clockwise')
        if name == 'mac_camera' and rotation != 0:
            raise ValueError('Mac camera must declare unrotated capture')
        if info.get('video_file', filename) != filename:
            raise ValueError(f'{name} video_file must be {filename}')
        if not (directory / filename).is_file():
            raise ValueError(f'missing required camera video: {directory / filename}')

        decoded = []
        try:
            for frame in video_frames(directory / filename):
                if ('width' in info and info['width'] != frame.shape[1]) or ('height' in info and info['height'] != frame.shape[0]):
                    raise ValueError(f'{name} video dimensions disagree with metadata')

                decoded.append(preprocess_image(frame, image_width, image_height, rotation))
        except av.error.FFmpegError as error:
            raise ValueError(f'cannot decode {directory / filename}: {error}') from error
        if not decoded:
            raise ValueError(f'{filename} contains no video frames')

        frames = torch.stack(decoded)
        if indices.max().item() >= frames.shape[0]:
            raise ValueError(f'{name} frame index references a nonexistent video frame')

        views.append(frames)
    paired_indices: Int64[Tensor, "T I"] = torch.stack((frame_indices, mac_indices), dim=1)
    return Episode(episode_id, metadata, table, (views[0], views[1]), paired_indices, positions, actions)
