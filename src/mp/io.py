from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

PathLike = Union[str, Path]


def save_json(obj: Dict[str, Any], path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json(path: PathLike) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_repo_root(start: str | Path | None = None) -> Path:
    p = Path(start) if start is not None else Path.cwd()
    p = p.resolve()

    for parent in [p, *p.parents]:
        if (parent / "pyproject.toml").exists():
            return parent

    raise RuntimeError(
        f"Could not find repo root (pyproject.toml) starting from: {p}"
    )