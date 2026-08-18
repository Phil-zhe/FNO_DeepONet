from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path.resolve())
    cfg["_project_root"] = str(path.resolve().parents[1])
    return cfg


def project_path(cfg: dict[str, Any], *parts: str) -> Path:
    return Path(cfg["_project_root"]).joinpath(*parts)


def resolve_path(cfg: dict[str, Any], path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_path(cfg, str(path))
