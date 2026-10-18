"""YAML configuration I/O shared by BC and future RL entry points."""

import json
from pathlib import Path
from typing import Any, Mapping

import yaml


def _plain_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a config into portable dict/list/scalar values for YAML and W&B."""
    if not isinstance(config, Mapping):
        raise ValueError("configuration must be a mapping")

    def validate(value: Any) -> None:
        if isinstance(value, Mapping):
            if not all(isinstance(key, str) for key in value):
                raise ValueError("configuration keys must be strings")

            for child in value.values():
                validate(child)
        elif isinstance(value, list):
            for child in value:
                validate(child)
        elif value is not None and not isinstance(value, (str, int, float, bool)):
            raise ValueError("configuration values must be mappings, lists or scalars")

    validate(config)
    try:
        return json.loads(json.dumps(dict(config), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("configuration must contain finite, serializable values") from error


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping without adding defaults or changing relative paths."""
    with Path(path).open() as file:
        config = yaml.safe_load(file)
    return _plain_config(config)


def save_config(config: Mapping[str, Any], path: str | Path) -> Path:
    """Save the caller's resolved run configuration; create parent directories."""
    path = Path(path)
    config = _plain_config(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path
