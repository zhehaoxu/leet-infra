#!/usr/bin/env python3
"""Validate bilingual content: description.en.md / notes.en.md / title_en.

Scans both the normal problem bank (problems/*/*/meta.json) and project
milestones (problems/projects/*/steps/*/meta.json), plus each project's
architecture.md / overview.md pages. The rules are identical everywhere.

Hard failures (exit 1): missing translation files, invalid meta.json, missing
title_en, heading-count mismatch, or fenced-code-count mismatch. Soft warnings:
leftover CJK anywhere in the file, and example/code lines whose content differs
between zh and en.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBLEMS = ROOT / "problems"
PROJECTS = PROBLEMS / "projects"
HAN = re.compile(r"[\u4e00-\u9fff]")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
FENCE = re.compile(r"^\s*```")


def headings(text: str) -> list[int]:
    return [len(m.group(1)) for m in (HEADING.match(line) for line in text.splitlines()) if m]


def fences(text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] | None = None
    for line in text.splitlines():
        if FENCE.match(line):
            if current is None:
                current = []
            else:
                blocks.append("\n".join(current))
                current = None
        elif current is not None:
            current.append(line)
    return blocks


LABEL = re.compile(r"^\s*(输入|输出|形状|返回|参数|示例|Input|Output|Shape|Returns|Args|Example)\s*[:：]")
# 示例块里的全角标点和空白会在翻译时变成半角/增减，不应算作内容改动
PUNCT = str.maketrans(dict(zip("，；：（）【】—－−～", ",;:()[]----~-")))


def norm_line(line: str) -> str:
    line = LABEL.sub("LABEL:", line.split("#", 1)[0]).translate(PUNCT)
    return re.sub(r"\s+", "", line.replace("→", "->"))


def block_diff(zh: str, en: str) -> str | None:
    """对比代码块；含中文的行属于待翻译 prose，只要求行结构一致。"""
    z = [line for line in zh.splitlines() if line.strip()]
    e = [line for line in en.splitlines() if line.strip()]
    if len(z) != len(e):
        return f"line count {len(z)} vs {len(e)}"
    for zl, el in zip(z, e):
        if HAN.search(zl):
            continue
        if norm_line(zl) != norm_line(el):
            return f"{zl.strip()[:40]!r} -> {el.strip()[:40]!r}"
    return None


def check_pair(directory: Path, rel: Path, stem: str,
               errors: list[str], warnings: list[str]) -> int:
    """校验一对 zh/en Markdown；返回计入的已翻译文件数。"""
    source = directory / f"{stem}.md"
    target = directory / f"{stem}.en.md"
    if not source.exists():
        errors.append(f"{rel}: missing {stem}.md")
        return 0
    if not target.exists():
        errors.append(f"{rel}: missing {stem}.en.md")
        return 0
    zh = source.read_text(encoding="utf-8")
    en = target.read_text(encoding="utf-8")

    if headings(zh) != headings(en):
        errors.append(
            f"{rel}/{stem}.en.md: heading levels differ "
            f"({headings(zh)} vs {headings(en)})"
        )
    zh_fences, en_fences = fences(zh), fences(en)
    if len(zh_fences) != len(en_fences):
        errors.append(
            f"{rel}/{stem}.en.md: fenced blocks differ "
            f"({len(zh_fences)} vs {len(en_fences)})"
        )
    else:
        for i, (a, b) in enumerate(zip(zh_fences, en_fences), 1):
            diff = block_diff(a, b)
            if diff:
                warnings.append(f"{rel}/{stem}.en.md: code block {i} changed ({diff})")

    residual = sum(1 for line in en.splitlines() if HAN.search(line))
    if residual:
        warnings.append(f"{rel}/{stem}.en.md: {residual} line(s) still contain CJK")
    return 1


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    # 普通题（problems/<fw>/<slug>/meta.json）与项目步骤（projects/<slug>/steps/<NN>-*/meta.json）
    # 判定标准一致；项目侧单独扫描，不进普通题库的目录 glob 口径。
    normal_metas = sorted(PROBLEMS.glob("*/*/meta.json"))
    project_metas = sorted(PROJECTS.glob("*/steps/*/meta.json"))
    metas = normal_metas + project_metas
    translated = 0

    for meta_path in metas:
        directory = meta_path.parent
        rel = directory.relative_to(PROBLEMS)
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{rel}: invalid meta.json: {exc}")
            continue
        title_en = meta.get("title_en", "").strip()
        if not title_en:
            errors.append(f"{rel}: missing title_en")
            continue

        for stem in ("description", "notes"):
            translated += check_pair(directory, rel, stem, errors, warnings)

    # 项目级开篇页 / 总览页也要双语齐备（project.md §6.5）
    for project_dir in sorted(
        p for p in PROJECTS.iterdir()
        if p.is_dir() and not p.name.startswith(("_", "."))
    ):
        rel = project_dir.relative_to(PROBLEMS)
        for stem in ("architecture", "overview"):
            translated += check_pair(project_dir, rel, stem, errors, warnings)

    print(f"problems: {len(normal_metas)}, project steps: {len(project_metas)}, "
          f"translated files: {translated}")
    for warning in warnings:
        print(f"WARN  {warning}")
    for error in errors:
        print(f"ERROR {error}")
    print(f"summary: {len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
