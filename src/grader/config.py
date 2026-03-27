"""Configuration loading and merging for the Robot Grader system."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    config_dir: Path | str | None = None,
    fruit_type: str | None = None,
) -> dict[str, Any]:
    """Load default config and optionally merge a fruit-type profile.

    Parameters
    ----------
    config_dir:
        Directory containing ``default.yaml`` and ``fruit_profiles/``.
        Defaults to the repo's ``config/`` directory.
    fruit_type:
        If provided, the matching ``fruit_profiles/<fruit_type>.yaml`` is
        deep-merged on top of the defaults.  If *None*, the ``fruit_type``
        key inside ``default.yaml`` is used.
    """
    config_dir = Path(config_dir) if config_dir else _DEFAULT_CONFIG_DIR

    default_path = config_dir / "default.yaml"
    with open(default_path) as f:
        cfg = yaml.safe_load(f)

    fruit = fruit_type or cfg.get("fruit_type")
    if fruit:
        profile_path = config_dir / "fruit_profiles" / f"{fruit}.yaml"
        if profile_path.exists():
            with open(profile_path) as f:
                profile = yaml.safe_load(f) or {}
            cfg = _deep_merge(cfg, profile)
        cfg["fruit_type"] = fruit

    return cfg
