"""合集（collection）定义：从 problems/collections.json 读取，按 mtime 热加载。

每个合集 = 名称 + 描述 + 有序的题目 slug 列表；题目可以同时出现在多个合集中。
不存在的 slug 在加载时被过滤掉，题库删题不会导致合集接口报错。
"""

from __future__ import annotations

import json

from .i18n import normalize
from .problems import PROBLEMS_DIR, all_problems

COLLECTIONS_FILE = PROBLEMS_DIR / "collections.json"

_CACHE: tuple[float, list[dict]] | None = None


def _load() -> list[dict]:
    if not COLLECTIONS_FILE.exists():
        return []
    raw = json.loads(COLLECTIONS_FILE.read_text(encoding="utf-8"))
    valid = {p.slug for p in all_problems()}
    result = []
    for item in raw:
        problems = [s for s in item.get("problems", []) if s in valid]
        result.append(
            {
                "slug": item["slug"],
                "name": item.get("name", item["slug"]),
                "name_en": item.get("name_en", ""),
                "description": item.get("description", ""),
                "description_en": item.get("description_en", ""),
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


def localized_name(collection: dict, lang: str | None = "zh") -> str:
    if normalize(lang) == "en" and collection.get("name_en"):
        return collection["name_en"]
    return collection["name"]


def localized_description(collection: dict, lang: str | None = "zh") -> str:
    if normalize(lang) == "en" and collection.get("description_en"):
        return collection["description_en"]
    return collection["description"]


def collections_of(problem_slug: str, lang: str | None = "zh") -> list[dict]:
    """反查题目所属合集（保持合集配置顺序）。"""
    return [
        {"slug": c["slug"], "name": localized_name(c, lang)}
        for c in all_collections()
        if problem_slug in c["problems"]
    ]
