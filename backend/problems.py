"""Problem bank loading: scan problems/<framework>/<slug>/ directories.

Each problem declares implementation variants in meta.json (default: single
``pytorch`` variant). Files live under ``variants/<key>/`` (starter.py,
test_cases.py, solution.py); unavailable variants are declared only, no dir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .i18n import normalize

ROOT = Path(__file__).resolve().parent.parent
PROBLEMS_DIR = ROOT / "problems"

DEFAULT_VARIANTS = {"pytorch": {"label": "PyTorch", "available": True}}


@dataclass
class VariantInfo:
    key: str
    label: str
    available: bool

    def as_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "available": self.available}


@dataclass
class Problem:
    slug: str
    framework: str
    title: str
    difficulty: str
    tags: list[str] = field(default_factory=list)
    order: int = 0
    variants: list[VariantInfo] = field(default_factory=list)
    title_en: str = ""

    @property
    def directory(self) -> Path:
        return PROBLEMS_DIR / self.framework / self.slug

    @property
    def default_variant(self) -> str:
        for v in self.variants:
            if v.available:
                return v.key
        return self.variants[0].key

    def resolve_variant(self, key: str | None) -> str:
        """Unset/unavailable/unknown variants fall back to the default one."""
        if key:
            for v in self.variants:
                if v.key == key and v.available:
                    return v.key
        return self.default_variant

    def _variant_dir(self, variant: str | None) -> Path:
        return self.directory / "variants" / (variant or self.default_variant)

    def localized_title(self, lang: str | None = "zh") -> str:
        if normalize(lang) == "en" and self.title_en:
            return self.title_en
        return self.title

    def _localized_markdown(self, stem: str, lang: str | None) -> str:
        """Prefer <stem>.en.md for English; fall back to the Chinese source."""
        if normalize(lang) == "en":
            en = self.directory / f"{stem}.en.md"
            if en.exists():
                return en.read_text(encoding="utf-8")
        p = self.directory / f"{stem}.md"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def description(self, lang: str | None = "zh") -> str:
        return self._localized_markdown("description", lang)

    def notes(self, lang: str | None = "zh") -> str:
        # 可选的题目详解笔记：与 description.md 同级；不存在则为空（前端据此置灰笔记标签）
        return self._localized_markdown("notes", lang)

    def starter(self, variant: str | None = None) -> str:
        return (self._variant_dir(variant) / "starter.py").read_text(encoding="utf-8")

    def test_cases(self, variant: str | None = None) -> str:
        return (self._variant_dir(variant) / "test_cases.py").read_text(encoding="utf-8")

    def solution(self, variant: str | None = None) -> str:
        return (self._variant_dir(variant) / "solution.py").read_text(encoding="utf-8")

    def summary(self, solved: bool, lang: str | None = "zh") -> dict:
        return {
            "slug": self.slug,
            "framework": self.framework,
            "title": self.localized_title(lang),
            "difficulty": self.difficulty,
            "tags": self.tags,
            "solved": solved,
            "variants": [v.as_dict() for v in self.variants],
        }


def _parse_variants(meta: dict) -> list[VariantInfo]:
    raw = meta.get("variants") or DEFAULT_VARIANTS
    return [
        VariantInfo(key=k, label=v.get("label", k), available=bool(v.get("available", True)))
        for k, v in raw.items()
    ]


def load_problems() -> list[Problem]:
    problems: list[Problem] = []
    for meta_path in sorted(PROBLEMS_DIR.glob("*/*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        problems.append(
            Problem(
                slug=meta_path.parent.name,
                framework=meta_path.parent.parent.name,
                title=meta["title"],
                difficulty=meta["difficulty"],
                tags=meta.get("tags", []),
                order=meta.get("order", 0),
                variants=_parse_variants(meta),
                title_en=meta.get("title_en", ""),
            )
        )
    problems.sort(key=lambda p: (p.framework, p.order, p.slug))
    return problems


_INDEX: tuple[float, dict[str, Problem]] | None = None


def _fingerprint() -> float:
    """题库目录的 mtime 指纹，用于热加载。

    覆盖：problems 根目录、各框架目录（增删题目目录时其 mtime 会变）、
    各题 meta.json（内容更新时 mtime 会变）。
    """
    stamps = [PROBLEMS_DIR.stat().st_mtime]
    for framework_dir in PROBLEMS_DIR.glob("*/"):
        stamps.append(framework_dir.stat().st_mtime)
        for meta in framework_dir.glob("*/meta.json"):
            stamps.append(meta.stat().st_mtime)
    return max(stamps)


def _ensure_index() -> dict[str, Problem]:
    global _INDEX
    fingerprint = _fingerprint()
    if _INDEX is None or _INDEX[0] != fingerprint:
        _INDEX = (fingerprint, {p.slug: p for p in load_problems()})
    return _INDEX[1]


def get_problem(slug: str) -> Problem | None:
    return _ensure_index().get(slug)


def all_problems() -> list[Problem]:
    """按题库顺序返回全部题目；与 get_problem 共用同一份 mtime 缓存。"""
    return list(_ensure_index().values())
