"""Local single-user progress store backed by data/progress.json."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "progress.json"

_LOCK = threading.Lock()


def _read() -> dict:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"solved": {}}


def _write(store: dict) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = DATA_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(DATA_FILE)


def is_solved(slug: str) -> bool:
    with _LOCK:
        return slug in _read()["solved"]


def mark_solved(slug: str) -> None:
    with _LOCK:
        store = _read()
        if slug not in store["solved"]:
            store["solved"][slug] = datetime.now(timezone.utc).isoformat()
            _write(store)


def solved_map() -> dict[str, str]:
    with _LOCK:
        return dict(_read()["solved"])
