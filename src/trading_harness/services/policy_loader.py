from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: str) -> dict[str, Any]:
    resolved = Path(path)
    if resolved.is_absolute() and not resolved.exists():
        # Container paths start with /app/; strip /app prefix
        parts = resolved.parts
        if len(parts) >= 2 and parts[1] == "app":
            fallback = _REPO_ROOT / Path(*parts[2:])
            if fallback.exists():
                resolved = fallback
    if not resolved.is_absolute():
        resolved = _REPO_ROOT / resolved
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Policy {path} must contain a YAML object")
    return data
