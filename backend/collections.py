"""合集（collection）定义：从 problems/collections.json 读取，按 mtime 热加载。

每个合集 = 名称 + 描述 + 有序的题目 slug 列表；题目可以同时出现在多个合集中。
不存在的 slug 在加载时被过滤掉，题库删题不会导致合集接口报错。
"""

from __future__ import annotations

import json

from .problems import PROBLEMS_DIR, load_problems

COLLECTIONS_FILE = PROBLEMS_DIR / "collections.json"

_CACHE: tuple[float, list[dict]] | None = None


def _load() -> list[dict]:
    if not COLLECTIONS_FILE.exists():
        return []
    raw = json.loads(COLLECTIONS_FILE.read_text(encoding="utf-8"))
    valid = {p.slug for p in load_problems()}
    result = []
    for item in raw:
        problems = [s for s in item.get("problems", []) if s in valid]
        result.append(
            {
                "slug": item["slug"],
                "name": item.get("name", item["slug"]),
                "description": item.get("description", ""),
                "problems": problems,
            }
        )
    return result


def all_collections() -> list[dict]:
    global _CACHE
    mtime = COLLECTIONS_FILE.stat().st_mtime if COLLECTIONS_FILE.exists() else 0.0
    if _CACHE is None or _CACHE[0] != mtime:
        _CACHE = (mtime, _load())
    return _CACHE[1]


def get_collection(slug: str) -> dict | None:
    for c in all_collections():
        if c["slug"] == slug:
            return c
    return None
