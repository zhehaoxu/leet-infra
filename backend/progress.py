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


def unmark(slug: str) -> bool:
    """取消单题已解记录（重刷粒度），返回是否确实移除。"""
    with _LOCK:
        store = _read()
        if slug not in store["solved"]:
            return False
        del store["solved"][slug]
        _write(store)
        return True


def solved_map() -> dict[str, str]:
    with _LOCK:
        return dict(_read()["solved"])


def reset() -> None:
    """清空全部已解决记录（用于二刷；不动草稿）。"""
    with _LOCK:
        _write({"solved": {}})


def export_store() -> dict:
    """导出当前进度（与 data/progress.json 同构）。"""
    with _LOCK:
        return {"solved": dict(_read()["solved"])}


def replace_all(solved: dict[str, str]) -> None:
    """全量覆盖（导入恢复场景），锁内原子写。"""
    with _LOCK:
        _write({"solved": dict(solved)})
