"""The mission's state, left in a file for whatever shows it (tools/panel.py).

The mission runner writes it as it changes; a panel in another window reads
it. Written whole to a file beside it and moved over, so a reader never
sees half of one.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Final

__all__ = ["STATUS_FILE", "read_status", "write_status"]

STATUS_FILE: Final = Path(__file__).resolve().parents[2] / "data" / "status.json"


def write_status(path: Path, state: str, message: str, **extra: object) -> None:
    """The state now, its message, and anything else worth showing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = path.with_name(path.name + ".new")
    data = {"state": state, "message": message, "at": time.time(), **extra}
    fresh.write_text(json.dumps(data, default=str), encoding="utf-8")
    fresh.replace(path)


def read_status(path: Path) -> dict | None:
    """The last state written; None where there is none, or it is unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
