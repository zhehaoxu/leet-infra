"""Project collections: an independent registry, ledger, and safe path resolver.

Projects deliberately live outside ``problems/pytorch/`` and outside
``collections.json`` (project.md section 1). Every filesystem path built here is
resolved through ``_safe`` so that a slug / step / variant / module taken from a
URL can never escape ``problems/projects/``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .i18n import normalize

ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = ROOT / "problems" / "projects"
REGISTRY = PROJECTS_DIR / "projects.json"

SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
STEP_DIR_RE = re.compile(r"^\d{2}-[a-z0-9-]+$")
VARIANT_RE = re.compile(r"^[a-z0-9_]+$")
MODULE_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
PAGE_RE = re.compile(r"^[a-z_]+\.md$")

STEP_ROOT_FILES = ("meta.json", "description.md", "description.en.md",
                   "notes.md", "notes.en.md")


class ProjectError(Exception):
    """Invalid project reference (bad slug, unknown step, or path traversal)."""


def _safe(base: Path, name: str, pattern: re.Pattern, kind: str) -> Path:
    """Join ``base`` and ``name`` only if the result stays inside ``base``."""
    if not isinstance(name, str) or not pattern.fullmatch(name):
        raise ProjectError(f"invalid {kind}: {name!r}")
    root = base.resolve()
    target = (root / name).resolve()
    if not target.is_relative_to(root) or target == root:
        raise ProjectError(f"{kind} escapes its root: {name!r}")
    return target


def _read_json(path: Path) -> dict | list:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProjectError(f"missing file: {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise ProjectError(f"invalid JSON: {path.name}: {exc}") from exc


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ProjectError(f"missing file: {path.name}") from exc


def _markdown(directory: Path, filename: str, lang: str | None) -> str:
    """English prefers <stem>.en.md and falls back to the Chinese source."""
    if not PAGE_RE.fullmatch(filename):
        raise ProjectError(f"invalid page name: {filename!r}")
    stem = filename[:-3]
    if normalize(lang) == "en":
        en = _safe(directory, f"{stem}.en.md", re.compile(r"^[a-z_]+\.en\.md$"), "page")
        if en.exists():
            return _read_text(en)
    return _read_text(_safe(directory, filename, PAGE_RE, "page"))


def _localized(entry: dict, key: str, lang: str | None) -> str:
    if normalize(lang) == "en":
        return entry.get(f"{key}_en") or entry.get(key, "")
    return entry.get(key, "")


def all_projects() -> list[dict]:
    """Read the display-level registry; the frontend tab reads only this file."""
    if not REGISTRY.exists():
        return []
    data = _read_json(REGISTRY)
    if not isinstance(data, list):
        raise ProjectError("projects.json must be a list")
    for entry in data:
        if not isinstance(entry, dict) or not SLUG_RE.fullmatch(str(entry.get("slug", ""))):
            raise ProjectError("projects.json contains an invalid slug")
    return sorted(data, key=lambda e: (e.get("display_order", 0), e.get("slug", "")))


def get_project(slug: str) -> tuple[Path, dict] | None:
    """Return (project directory, project.json) or None for an unknown slug."""
    directory = _safe(PROJECTS_DIR, slug, SLUG_RE, "project slug")
    if not directory.is_dir():
        return None
    ledger = _read_json(directory / "project.json")
    if not isinstance(ledger, dict) or ledger.get("slug") != slug:
        raise ProjectError("project.json slug does not match its directory")
    return directory, ledger


def ordered_steps(ledger: dict) -> list[dict]:
    steps = ledger.get("steps", [])
    if not isinstance(steps, list):
        raise ProjectError("project.json steps must be a list")
    return sorted(steps, key=lambda s: int(s["step"]))


def step_number(value) -> int:
    """URL step numbers are ints in [1, 99]; anything else is a bad reference."""
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ProjectError(f"invalid step number: {value!r}") from exc
    if number < 1 or number > 99:
        raise ProjectError(f"step out of range: {number}")
    return number


def get_step(slug: str, step: int) -> tuple[Path, dict, dict, dict]:
    """Resolve one milestone: (step directory, ledger, step ledger row, meta)."""
    found = get_project(slug)
    if found is None:
        raise ProjectError(f"unknown project: {slug}")
    directory, ledger = found
    number = step_number(step)
    rows = [s for s in ordered_steps(ledger) if int(s["step"]) == number]
    if not rows:
        raise ProjectError(f"unknown step: {step}")
    row = rows[0]
    folder = str(row.get("slug", ""))
    if not folder.startswith(f"{number:02d}-"):
        raise ProjectError(f"step folder prefix mismatch: {folder}")
    step_dir = _safe(directory / "steps", folder, STEP_DIR_RE, "step folder")
    if not step_dir.is_dir():
        raise ProjectError(f"missing step directory: {folder}")
    meta = _read_json(step_dir / "meta.json")
    if not isinstance(meta, dict):
        raise ProjectError("step meta.json must be an object")
    return step_dir, ledger, row, meta


def variant_dir(step_dir: Path, variant: str | None, meta: dict) -> Path:
    variants = meta.get("variants") or {"pytorch": {"available": True}}
    key = variant or next(
        (k for k, v in variants.items() if isinstance(v, dict) and v.get("available", True)),
        "pytorch",
    )
    if key not in variants:
        key = "pytorch"
    return _safe(step_dir / "variants", key, VARIANT_RE, "variant")


def scaffold_sources(project_dir: Path, imports: list[str]) -> dict[str, str]:
    """Load the frozen official modules declared by this step, in dependency order."""
    base = project_dir / "scaffold"
    sources: dict[str, str] = {}
    for name in imports or []:
        if not MODULE_RE.fullmatch(name):
            raise ProjectError(f"invalid scaffold module: {name!r}")
        path = _safe(base, f"{name}.py", re.compile(r"^[a-z_][a-z0-9_]*\.py$"), "scaffold module")
        sources[name] = _read_text(path)
    return sources


def summary_list(lang: str | None, progress_of) -> list[dict]:
    """Display-level list for the frontend tab, enriched with progress counts."""
    result = []
    for entry in all_projects():
        slug = entry["slug"]
        prog = progress_of(slug)
        result.append({
            "slug": slug,
            "name": _localized(entry, "name", lang),
            "summary": _localized(entry, "summary", lang),
            "difficulty": entry.get("difficulty", "Medium"),
            "status": entry.get("status", "draft"),
            "step_count": entry.get("step_count", 0),
            "display_order": entry.get("display_order", 0),
            "completed_steps": len(prog.get("completed_steps", [])),
        })
    return result


def detail(slug: str, lang: str | None, progress_of) -> dict:
    """Full project page: registry row + ledger rows + progress + pages."""
    found = get_project(slug)
    if found is None:
        raise ProjectError(f"unknown project: {slug}")
    directory, ledger = found
    entry = next((e for e in all_projects() if e["slug"] == slug), {})
    prog = progress_of(slug)
    completed = set(prog.get("completed_steps", []))
    steps = []
    for row in ordered_steps(ledger):
        number = int(row["step"])
        folder = str(row.get("slug", ""))
        step_dir = _safe(directory / "steps", folder, STEP_DIR_RE, "step folder")
        meta = _read_json(step_dir / "meta.json")
        steps.append({
            "step": number,
            "slug": folder,
            "title": _localized(row, "title", lang) or meta.get("title", ""),
            "title_en": row.get("title_en", ""),
            "why": _localized(row, "why", lang),
            "depends_on": row.get("depends_on", []),
            "verification": row.get("verification", []),
            "scaffold_imports": row.get("scaffold_imports", []),
            "final_step": bool(row.get("final_step", False)),
            "status": row.get("status", "draft"),
            "difficulty": meta.get("difficulty", "Medium"),
            "tags": meta.get("tags", []),
            "completed": number in completed,
        })
    unlocked = bool(prog.get("unlocked_overview", False))
    return {
        "slug": slug,
        "name": _localized(entry or ledger, "name", lang),
        "summary": _localized(entry, "summary", lang),
        "difficulty": entry.get("difficulty", "Medium"),
        "status": entry.get("status", "draft"),
        "step_count": len(steps),
        "framework": ledger.get("framework", "pytorch"),
        "scaffold_version": ledger.get("scaffold_version", ""),
        "architecture": _markdown(directory, ledger.get("architecture_page", "architecture.md"), lang),
        "overview": (
            _markdown(directory, ledger.get("overview_page", "overview.md"), lang)
            if unlocked else None
        ),
        "overview_unlocked": unlocked,
        "overview_unlock_step": ledger.get("overview_unlock_step"),
        "review_step": ledger.get("review_step"),
        "progress": prog,
        "steps": steps,
    }


def step_detail(slug: str, step: int, lang: str | None, completed: bool) -> dict:
    """Workbench payload for one milestone."""
    step_dir, ledger, row, meta = get_step(slug, step)
    vdir = variant_dir(step_dir, None, meta)
    numbers = [int(s["step"]) for s in ordered_steps(ledger)]
    index = numbers.index(step)
    return {
        "project": {
            "slug": slug,
            "name": ledger.get("name", ""),
            "name_en": ledger.get("name_en", ""),
        },
        "step": {
            "step": step,
            "slug": row.get("slug", ""),
            "title": _localized(row, "title", lang) or meta.get("title", ""),
            "difficulty": meta.get("difficulty", "Medium"),
            "why": _localized(row, "why", lang),
            "depends_on": row.get("depends_on", []),
            "verification": row.get("verification", []),
            "scaffold_imports": row.get("scaffold_imports", []),
            "final_step": bool(row.get("final_step", False)),
            "completed": completed,
        },
        "description": _markdown(step_dir, "description.md", lang),
        "notes": _markdown(step_dir, "notes.md", lang),
        "starter_code": _read_text(vdir / "starter.py"),
        "active_variant": vdir.name,
        "total_steps": len(numbers),
        "prev_step": numbers[index - 1] if index > 0 else None,
        "next_step": numbers[index + 1] if index + 1 < len(numbers) else None,
    }
