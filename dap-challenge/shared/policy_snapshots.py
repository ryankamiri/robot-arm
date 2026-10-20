"""Portable in-memory actor snapshots shared by learners and live inference."""

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import pickle
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import torch

from models.policy import DapPolicy


MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class ActorSnapshot:
    """Immutable wire bytes plus publication identity; never alias trainable parameters."""

    content: bytes
    run_id: str
    actor_updates: int
    digest: str

    @property
    def version(self) -> str:
        return f'{self.run_id}/actor-{self.actor_updates:09d} | {self.digest[:12]}'


def encode_actor(policy: DapPolicy, preprocessing: dict, run_id: str, actor_updates: int) -> ActorSnapshot:
    """Clone a consistent actor after an optimizer round into an independent CPU payload."""
    payload = {'format_version': 1, 'run_id': run_id, 'actor_updates': actor_updates,
               'model': {'history_length': policy.history_length, 'log_std_min': policy.log_std_min,
                         'log_std_max': policy.log_std_max}, 'preprocessing': dict(preprocessing),
               'weights': {key: value.detach().to(device='cpu', copy=True) for key, value in policy.state_dict().items()}}
    buffer = BytesIO()
    torch.save(payload, buffer)
    content = buffer.getvalue()
    return ActorSnapshot(content, run_id, actor_updates, hashlib.sha256(content).hexdigest())


def decode_actor(content: bytes) -> tuple[ActorSnapshot, DapPolicy, dict]:
    """Validate a CPU snapshot before it can enter the runtime command queue."""
    if not content or len(content) > MAX_SNAPSHOT_BYTES:
        raise ValueError('actor snapshot is empty or exceeds the 16 MiB limit')

    try:
        state = torch.load(BytesIO(content), map_location='cpu', weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError, IndexError, KeyError, ValueError) as error:
        raise ValueError('Invalid binary actor snapshot.') from error

    if not isinstance(state, dict) or state.get('format_version') != 1:
        raise ValueError('unsupported actor snapshot format')

    if not isinstance(state.get('run_id'), str) or not state['run_id'].strip() or len(state['run_id']) > 128:
        raise ValueError('snapshot requires a bounded nonempty run identity')

    step = state.get('actor_updates')
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError('snapshot actor update count must be a nonnegative integer')

    if state.get('preprocessing') != {'image_width': 224, 'image_height': 168}:
        raise ValueError('snapshot must use the live 224x168 preprocessing contract')

    model = state.get('model')
    if not isinstance(model, dict) or set(model) != {'history_length', 'log_std_min', 'log_std_max'} or model['history_length'] != 3:
        raise ValueError('snapshot must describe the live three-observation actor')

    weights = state.get('weights')
    if not isinstance(weights, dict) or any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32 for value in weights.values()):
        raise ValueError('snapshot weights must be float32 tensors')

    policy = DapPolicy(**model)
    policy.load_state_dict(weights, strict=True)
    if any(not torch.isfinite(value).all() for value in policy.state_dict().values()):
        raise ValueError('actor snapshot contains nonfinite weights')

    snapshot = ActorSnapshot(content, state['run_id'], step, hashlib.sha256(content).hexdigest())
    return snapshot, policy, {'model': model, 'preprocessing': state['preprocessing']}


def send_actor(snapshot: ActorSnapshot, url: str, timeout_s: float) -> dict:
    """Transfer bytes directly to the local inference service and verify its acknowledgement."""
    parsed = urlparse(url)
    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost'):
        raise ValueError('actor publication requires a local HTTP endpoint')

    request = Request(url, data=snapshot.content, headers={'Content-Type': 'application/octet-stream'}, method='POST')
    try:
        with urlopen(request, timeout=timeout_s) as response:
            acknowledgement = json.load(response)
    except HTTPError as error:
        raise RuntimeError(f'Actor publication rejected ({error.code}): {error.read(4096).decode(errors="replace")}') from error
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f'Actor publication failed: {error}') from error

    if not isinstance(acknowledgement, dict):
        raise ValueError('invalid inference publication acknowledgement')

    if acknowledgement.get('received_policy_version') != snapshot.version or acknowledgement.get('sha256') != snapshot.digest:
        raise ValueError('inference acknowledgement does not match the published actor')

    return acknowledgement
