"""Problem bank loading: scan problems/<framework>/<slug>/ directories."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBLEMS_DIR = ROOT / "problems"


@dataclass
class Problem:
    slug: str
    framework: str
    title: str
    difficulty: str
    tags: list[str] = field(default_factory=list)
    order: int = 0

    @property
    def directory(self) -> Path:
        return PROBLEMS_DIR / self.framework / self.slug

    def description(self) -> str:
        return (self.directory / "description.md").read_text(encoding="utf-8")

    def starter(self) -> str:
        return (self.directory / "starter.py").read_text(encoding="utf-8")

    def test_cases(self) -> str:
        return (self.directory / "test_cases.py").read_text(encoding="utf-8")

    def solution(self) -> str:
        return (self.directory / "solution.py").read_text(encoding="utf-8")

    def summary(self, solved: bool) -> dict:
        return {
            "slug": self.slug,
            "framework": self.framework,
            "title": self.title,
            "difficulty": self.difficulty,
            "tags": self.tags,
            "solved": solved,
        }


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


def get_problem(slug: str) -> Problem | None:
    global _INDEX
    fingerprint = _fingerprint()
    if _INDEX is None or _INDEX[0] != fingerprint:
        _INDEX = (fingerprint, {p.slug: p for p in load_problems()})
    return _INDEX[1].get(slug)
