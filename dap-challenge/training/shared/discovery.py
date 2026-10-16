"""Resolve remembered USB identities without opening cameras or motor ports."""

import json
from pathlib import Path
import subprocess
import sys

from serial.tools.list_ports import comports


def camera_inventory() -> list[dict]:
    if sys.platform != 'darwin':
        raise RuntimeError('Camera discovery currently supports macOS only.')

    result = subprocess.run(
        ['swift', str(Path(__file__).with_name('cameras.swift'))],
        capture_output=True, text=True, timeout=15, check=True,
    )
    return json.loads(result.stdout)


def device_inventory() -> dict:
    ports = [{'port': port.device, 'description': port.description, 'serial': port.serial_number}
             for port in comports()]
    try:
        return {'ports': ports, 'cameras': camera_inventory(), 'warnings': []}
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as error:
        return {'ports': ports, 'cameras': [], 'warnings': [f'Camera discovery failed: {error}']}


def resolve_profile(profile: dict, inventory: dict) -> tuple[dict, list[str]]:
    settings = {**profile}
    warnings = list(inventory['warnings'])
    for role in ('leader', 'follower'):
        serial = profile.get(f'{role}_serial')
        matches = [p for p in inventory['ports'] if serial and p['serial'] == serial]
        settings[f'{role}_port'] = matches[0]['port'] if len(matches) == 1 else ''
        if len(matches) != 1:
            warnings.append(f'{role.capitalize()} controller missing or ambiguous; select its port.')
    identity = profile.get('camera_device_id')
    matches = [c for c in inventory['cameras'] if identity and c['device_id'] == identity]
    settings['camera_index'] = matches[0]['index'] if len(matches) == 1 else None
    if len(matches) != 1:
        warnings.append('Saved camera missing or ambiguous; select a camera.')
    if settings['leader_port'] and settings['leader_port'] == settings['follower_port']:
        settings['leader_port'] = settings['follower_port'] = ''
        warnings.append('Leader and follower must be different controllers.')

    return settings, warnings


def remember_profile(path: Path, settings: dict, inventory: dict) -> None:
    profile = {key: value for key, value in settings.items() if key not in ('leader_port', 'follower_port', 'camera_index')}
    for role in ('leader', 'follower'):
        matches = [p for p in inventory['ports'] if p['port'] == settings[f'{role}_port']]
        profile[f'{role}_serial'] = matches[0]['serial'] if len(matches) == 1 else None
    matches = [c for c in inventory['cameras'] if c['index'] == settings['camera_index']]
    profile['camera_device_id'] = matches[0]['device_id'] if len(matches) == 1 else None
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(profile, indent=2) + '\n')
    temporary.replace(path)


def load_profile(path: Path) -> dict:
    if not path.exists():
        return {}

    return json.loads(path.read_text())


def resolve_connection(settings: dict, inventory: dict) -> dict:
    resolved = dict(settings)
    for role in ('leader', 'follower'):
        identity = settings.get(f'{role}_serial')
        if identity:
            matches = [p for p in inventory['ports'] if p['serial'] == identity]
            if len(matches) != 1:
                raise ValueError(f'{role.capitalize()} controller missing or ambiguous. Rediscover devices.')

            resolved[f'{role}_port'] = matches[0]['port']
    identity = settings.get('camera_device_id')
    if identity:
        matches = [c for c in inventory['cameras'] if c['device_id'] == identity]
        if len(matches) != 1:
            raise ValueError('Camera missing or ambiguous. Rediscover devices.')

        resolved['camera_index'] = matches[0]['index']
    if resolved['leader_port'] == resolved['follower_port']:
        raise ValueError('Leader and follower must use different controllers.')

    return resolved
