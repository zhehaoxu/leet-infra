"""参考解耗时基准缓存（data/benchmarks.json）。

题目首次 AC 后，后台线程用对应 variant 的参考 solution 全量跑一遍，缓存每个用例的
elapsed_ms。提交结果附带该基准供前端展示「你的耗时 vs 参考基准」——仅展示不设门槛。
存储结构：{slug: {variant: {"cases": {...}, "total_ms": ..., "measured_at": ...}}}。
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "benchmarks.json"

_LOCK = threading.Lock()


def _read() -> dict:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def get_benchmarks(slug: str, variant: str) -> dict[str, float] | None:
    """{case_name: ms}；无基准时返回 None。"""
    with _LOCK:
        entry = _read().get(slug)
    if not isinstance(entry, dict):
        return None
    by_variant = entry.get(variant)
    if not isinstance(by_variant, dict):
        return None
    cases = by_variant.get("cases")
    return cases if isinstance(cases, dict) and cases else None


def record(slug: str, variant: str, result: dict) -> bool:
    """从一次 accepted 的 run_judgement 结果提取用例耗时并落盘。"""
    cases = {
        c["name"]: c["elapsed_ms"]
        for c in result.get("cases", [])
        if c.get("passed") and isinstance(c.get("elapsed_ms"), (int, float))
    }
    if not cases:
        return False
    with _LOCK:
        store = _read()
        store.setdefault(slug, {})[variant] = {
            "cases": cases,
            "total_ms": result.get("elapsed_ms"),
            "measured_at": datetime.now(timezone.utc).isoformat(),
        }
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = DATA_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(DATA_FILE)
    return True
