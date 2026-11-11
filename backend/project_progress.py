"""Project progress: a store fully separate from data/progress.json.

Recorded by the frontend *after* a passing submission; the judge never writes
here, so a progress write failure cannot change a judging result (project.md
section 1.4).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "project_progress.json"

_LOCK = threading.Lock()


def _read() -> dict:
    if DATA_FILE.exists():
        try:
            data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return {}


def _write(store: dict) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = DATA_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(DATA_FILE)


def _blank() -> dict:
    return {"completed_steps": [], "unlocked_overview": False}


def for_project(slug: str) -> dict:
    with _LOCK:
        raw = _read().get(slug, {})
    if not isinstance(raw, dict):
        raw = {}
    return {
        "completed_steps": sorted(
            {int(s) for s in raw.get("completed_steps", []) if str(s).isdigit()}
        ),
        "unlocked_overview": bool(raw.get("unlocked_overview", False)),
    }


def all_progress() -> dict:
    with _LOCK:
        return {
            slug: for_project_unlocked(slug, data)
            for slug, data in _read().items()
        }


def for_project_unlocked(slug: str, raw: dict) -> dict:
    """Internal helper used while holding the lock."""
    if not isinstance(raw, dict):
        raw = {}
    return {
        "completed_steps": sorted(
            {int(s) for s in raw.get("completed_steps", []) if str(s).isdigit()}
        ),
        "unlocked_overview": bool(raw.get("unlocked_overview", False)),
    }


def is_completed(slug: str, step: int) -> bool:
    return step in for_project(slug)["completed_steps"]


def mark_step(slug: str, step: int, overview_unlock_step: int | None = None) -> dict:
    """Idempotently record a completed milestone; unlock the overview at the final step."""
    with _LOCK:
        store = _read()
        current = for_project_unlocked(slug, store.get(slug, {}))
        done = set(current["completed_steps"])
        done.add(int(step))
        unlocked = current["unlocked_overview"] or (
            overview_unlock_step is not None and int(step) == int(overview_unlock_step)
        )
        updated = {"completed_steps": sorted(done), "unlocked_overview": unlocked}
        store[slug] = updated
        _write(store)
        return updated


def unmark_step(slug: str, step: int) -> dict:
    """Redo one milestone; the overview stays unlocked (it is a cognitive asset)."""
    with _LOCK:
        store = _read()
        current = for_project_unlocked(slug, store.get(slug, {}))
        updated = {
            "completed_steps": [s for s in current["completed_steps"] if s != int(step)],
            "unlocked_overview": current["unlocked_overview"],
        }
        store[slug] = updated
        _write(store)
        return updated
