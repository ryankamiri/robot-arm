"""Versioned inference snapshots and bounded live observation histories."""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
import pickle
from typing import Any

from jaxtyping import Float
import torch
from torch import Tensor

from models.policy import DapPolicy
from models.distributions import sample_tanh_gaussian
from shared.devices import CameraFrame
from shared.normalization import JOINT_NAMES, denormalize_controller, normalize_controller
from shared.observations import preprocess_image


CHECKPOINT_NAMES = ('best_mae.pt', 'best.pt', 'latest.pt')


def checkpoint_path(root: Path, selection: str) -> Path:
    path = (root / selection).resolve()
    if not path.is_relative_to(root.resolve()) or path.name not in CHECKPOINT_NAMES or not path.is_file():
        raise ValueError('Select an existing checkpoint inside the canonical runs directory.')

    return path


def checkpoint_catalog(root: Path) -> dict[str, Any]:
    """List readable local training checkpoints without instantiating a policy."""
    items, warnings = [], []
    for path in root.glob('*/*.pt'):
        if path.name not in CHECKPOINT_NAMES:
            continue

        try:
            path = checkpoint_path(root, str(path.relative_to(root)))
            state = torch.load(path, map_location='cpu', weights_only=True)
            step = state['progress']['optimizer_step']
            run_name = state['config'].get('wandb', {}).get('name') or path.parent.name
            criterion = {'best_mae.pt': 'Best action MAE', 'best.pt': 'Best NLL', 'latest.pt': 'Latest'}[path.name]
            items.append({'path': str(path.relative_to(root.resolve())), 'label': f'{run_name} | {criterion} | step {step} | {path.parent.name}',
                          'step': step, 'modified': path.stat().st_mtime, 'kind': path.name})
        except (OSError, ValueError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
            warnings.append(f'{path.parent.name}/{path.name}: {error}')
    items.sort(key=lambda item: (item['kind'] != 'best_mae.pt', -item['modified']))
    return {'items': items, 'warnings': warnings}


@dataclass
class PolicyDecision:
    requested: list[float]
    normalized: list[float]
    version: str


class PolicyRunner:
    """Own an immutable actor copy; callers may construct snapshots from disk or memory."""

    def __init__(self, policy: DapPolicy, config: dict, version: str, device: str):
        self.device = torch.device(device)
        self.policy = deepcopy(policy).to(device=self.device, dtype=torch.float32).eval().requires_grad_(False)
        self.config = deepcopy(config)
        self.version = version
        self.history_length = policy.history_length
        # Bounded deques automatically discard the oldest observation when a full window receives a new one.
        self.images = deque(maxlen=self.history_length)
        self.positions = deque(maxlen=self.history_length)
        if self.history_length != 3 or self.config['preprocessing'] != {'image_width': 224, 'image_height': 168}:
            raise ValueError('Live inference requires the agreed three-observation, 224×168 input contract.')
        if any(not torch.isfinite(value).all() for value in self.policy.state_dict().values()):
            raise ValueError('Checkpoint contains nonfinite model weights.')

    @classmethod
    def from_checkpoint(cls, root: Path, selection: str, device: str):
        path = checkpoint_path(root, selection)
        content = path.read_bytes()
        state = torch.load(BytesIO(content), map_location='cpu', weights_only=True)
        if state.get('format_version') != 1 or set(state['models']) != {'actor'}:
            raise ValueError('Unsupported actor checkpoint.')

        policy = DapPolicy(**state['config']['model'])
        policy.load_state_dict(state['models']['actor'])
        version = f'{selection} | step {state["progress"]["optimizer_step"]} | {hashlib.sha256(content).hexdigest()[:12]}'
        return cls(policy, state['config'], version, device)

    def reset(self) -> None:
        self.images.clear()
        self.positions.clear()

    def observe(self, arm: CameraFrame, mac: CameraFrame, positions: list[float], rotation_clockwise: int) -> None:
        views: Float[Tensor, 'I C_in H_in W_in'] = torch.stack([
            preprocess_image(torch.from_numpy(arm.rgb), 224, 168, rotation_clockwise),
            preprocess_image(torch.from_numpy(mac.rgb), 224, 168, 0),
        ])
        normalized: Float[Tensor, 'A'] = normalize_controller(torch.tensor(positions, dtype=torch.float32), JOINT_NAMES)
        if not self.images:
            for _ in range(self.history_length - 1):
                self.images.append(views)
                self.positions.append(normalized)
        self.images.append(views)
        self.positions.append(normalized)

    def predict(self, mode: str) -> PolicyDecision:
        if mode not in ('mean', 'sample') or len(self.images) != self.history_length:
            raise ValueError('Choose mean/sample and populate live observation history first.')

        images: Float[Tensor, 'B L I C_in H_in W_in'] = torch.stack(list(self.images)).unsqueeze(0).to(self.device)
        positions: Float[Tensor, 'B L A'] = torch.stack(list(self.positions)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            parameters = self.policy(images, positions)
            if not torch.isfinite(parameters.mean).all() or not torch.isfinite(parameters.log_std).all():
                raise ValueError('Policy produced nonfinite Gaussian parameters.')

            if mode == 'mean':
                action: Float[Tensor, 'B A'] = parameters.mean.tanh()
            else:
                action = sample_tanh_gaussian(parameters).actions
            normalized: Float[Tensor, 'B A'] = action.cpu()
            targets: Float[Tensor, 'B A'] = denormalize_controller(normalized, JOINT_NAMES)
        return PolicyDecision(targets[0].tolist(), normalized[0].tolist(), self.version)
