"""Configuration loading (YAML with single-level inheritance through a ``base`` key)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from oilcast.paths import PROJECT_ROOT

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

Config = dict[str, Any]


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path | None = None) -> Config:
    """Load a YAML config; a ``base`` entry is resolved relative to the file and merged first."""
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = _deep_merge(load_config(path.parent / base), cfg)
    return cfg


def horizon_config(cfg: Config, hspec: dict) -> Config:
    """Config of one production horizon: its own input variables and linear shortcut (ronda 4).

    ``hspec`` is a horizon entry of the production spec. Without ``features``/``highway`` keys
    the horizon uses the global configuration unchanged.
    """
    out = cfg
    if hspec.get("features"):
        out = {**out, "features": _deep_merge(out["features"], hspec["features"])}
    if "highway" in hspec:
        out = {**out, "model": {**out.get("model", {}), "highway": bool(hspec["highway"])}}
    return out


def horizon_overrides(cfg: Config, h: int) -> dict:
    """``horizon_overrides`` entry of horizon ``h`` (``features`` and ``highway``), or {}."""
    overrides = cfg.get("horizon_overrides") or {}
    return overrides.get(str(h)) or overrides.get(h) or {}


def for_horizon(cfg: Config, h: int) -> Config:
    """Config used to train and evaluate horizon ``h`` (``horizon_overrides`` in the YAML)."""
    return horizon_config(cfg, horizon_overrides(cfg, h))
