"""Small configuration helpers for CLI workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a workflow configuration file is invalid."""


def load_yaml_config(config_path: Path | None) -> dict[str, Any]:
    """Load a YAML mapping or return an empty config when no path is provided."""
    if config_path is None:
        return {}
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"Config must be a YAML mapping: {config_path}")
    return loaded


def path_config_value(config: dict[str, Any], key: str, default: Path) -> Path:
    """Return a Path config value."""
    value = config.get(key, default)
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    raise ConfigError(f"Config field {key!r} must be a path string")


def int_config_value(config: dict[str, Any], key: str, default: int) -> int:
    """Return an integer config value."""
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"Config field {key!r} must be an integer")
    return value


def nested_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a nested mapping config value."""
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"Config field {key!r} must be a mapping")
    return value
