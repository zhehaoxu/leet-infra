"""项目层完整性护栏：独立文件，不塞进 test_problem_integrity.py。

纯读文件系统、零运行时依赖（不导入 backend / torch / judge）。项目题与普通
题库物理隔离、记录隔离（project.md §1 / §6.5），本文件兜住项目这一层自己
最容易漂移的结构：注册表与台账一致性、step 编号/前缀/依赖闭包、脚手架引用、
五件套与双语齐备、双语开篇/总览页、脚手架无隐藏状态、不混入普通合集。
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROJECTS_DIR = ROOT / "problems" / "projects"
REGISTRY = PROJECTS_DIR / "projects.json"
COLLECTIONS_FILE = ROOT / "problems" / "collections.json"

HAN = re.compile(r"[\u4e00-\u9fff]")
HEADING = re.compile(r"^(#{1,6})\s+")
FENCE = re.compile(r"^\s*```")

ROOT_FILES = ("meta.json", "description.md", "description.en.md", "notes.md", "notes.en.md")
VARIANT_FILES = ("starter.py", "solution.py", "test_cases.py")


def _registry():
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def _ledger(slug):
    return json.loads((PROJECTS_DIR / slug / "project.json").read_text(encoding="utf-8"))


def _project_slugs():
    return sorted(
        d.name for d in PROJECTS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(("_", "."))
    )


def _step_dir(project_dir, row):
    return project_dir / "steps" / row["slug"]


def _headings(text):
    return [len(m.group(1)) for m in (HEADING.match(line) for line in text.splitlines()) if m]


def _fence_count(text):
    return sum(1 for line in text.splitlines() if FENCE.match(line))


PROJECT_SLUGS = [e["slug"] for e in _registry()]
PROJECT_DIRS = _project_slugs()
STEP_CASES = [
    (slug, row["slug"])
    for slug in PROJECT_SLUGS
    for row in _ledger(slug)["steps"]
]


def test_registry_matches_project_ledgers():
    """(1) projects.json 与每个 project.json 的 slug/name/step_count 一致，slug 无重复。"""
    assert PROJECT_SLUGS, "projects.json 为空，护栏会假绿"
    assert len(PROJECT_SLUGS) == len(set(PROJECT_SLUGS)), "注册表中 slug 重复"
    assert sorted(PROJECT_SLUGS) == PROJECT_DIRS, "注册表条目与磁盘项目目录不一致"
    for entry in _registry():
        ledger = _ledger(entry["slug"])
        assert ledger["slug"] == entry["slug"]
        assert ledger["name"] == entry["name"]
        assert ledger.get("name_en") == entry.get("name_en")
        assert entry["step_count"] == len(ledger["steps"]), (
            f"{entry['slug']}: step_count={entry['step_count']} 与 steps 长度不符"
        )
        assert entry["status"] == ledger["status"], f"{entry['slug']}: status 不一致"


def test_registry_has_bilingual_display_info():
    for entry in _registry():
        for key in ("name", "name_en", "summary", "summary_en", "difficulty", "status"):
            assert str(entry.get(key, "")).strip(), f"{entry['slug']}: 缺少 {key}"
        assert isinstance(entry.get("display_order"), int)
        assert entry["status"] in ("draft", "live")


@pytest.mark.parametrize("slug", PROJECT_SLUGS)
def test_steps_sequential_prefix_and_dependencies(slug):
    """(2) step 从 1 连续、NN 前缀一致、depends_on 只指向更小的 step。"""
    ledger = _ledger(slug)
    steps = ledger["steps"]
    numbers = [int(s["step"]) for s in steps]
    assert numbers == list(range(1, len(steps) + 1)), f"{slug}: step 不连续: {numbers}"
    by_number = {int(s["step"]): s for s in steps}
    for row in steps:
        number = int(row["step"])
        assert row["slug"].startswith(f"{number:02d}-"), f"{slug}/{row['slug']}: NN 前缀不符"
        assert _step_dir(PROJECTS_DIR / slug, row).is_dir(), f"{slug}/{row['slug']}: 缺目录"
        for dep in row.get("depends_on", []):
            assert isinstance(dep, int) and 1 <= dep < number, (
                f"{slug}/{row['slug']}: 非法 depends_on={dep}"
            )
            assert dep in by_number, f"{slug}/{row['slug']}: 依赖了不存在的 step {dep}"


@pytest.mark.parametrize("slug", PROJECT_SLUGS)
def test_final_step_and_review_step(slug):
    """(2) final_step 唯一；L4 复盘作为 review_step 时可挂在终局之后（project.md §1.3）。"""
    ledger = _ledger(slug)
    steps = ledger["steps"]
    numbers = [int(s["step"]) for s in steps]
    finals = [int(s["step"]) for s in steps if s.get("final_step")]
    assert len(finals) == 1, f"{slug}: final_step 应唯一，实际 {finals}"
    review = ledger.get("review_step")
    if review is not None:
        assert review == max(numbers), f"{slug}: review_step 必须是最大 step"
        assert review > finals[0], f"{slug}: review_step 应在终局之后"
        assert ledger.get("overview_unlock_step") == finals[0], (
            f"{slug}: 总览应在终局步骤（{finals[0]}）通过后解锁"
        )
        by_number = {int(s["step"]): s for s in steps}
        assert by_number[review]["verification"] == ["L4"], "复盘步骤应只标 L4"
    else:
        assert finals[0] == max(numbers), f"{slug}: 无复盘步骤时终局必须是最大 step"


@pytest.mark.parametrize("slug", PROJECT_SLUGS)
def test_scaffold_imports_match_dependency_closure(slug):
    """(3) scaffold_imports 与 depends_on 传递闭包一致，且模块在 scaffold/ 存在。"""
    project_dir = PROJECTS_DIR / slug
    ledger = _ledger(slug)
    steps = ledger["steps"]
    by_number = {int(s["step"]): s for s in steps}
    module_of = {int(s["step"]): s.get("module") for s in steps}
    for row in steps:
        closure, stack = set(), list(row.get("depends_on", []))
        while stack:
            n = stack.pop()
            if n in closure:
                continue
            closure.add(n)
            stack.extend(by_number[n].get("depends_on", []))
        expected = {module_of[n] for n in closure if module_of.get(n)}
        actual = set(row.get("scaffold_imports", []))
        assert actual == expected, (
            f"{slug}/{row['slug']}: scaffold_imports={sorted(actual)} 与依赖闭包 "
            f"{sorted(expected)} 不一致"
        )
        for name in row.get("scaffold_imports", []):
            path = project_dir / "scaffold" / f"{name}.py"
            assert path.is_file(), f"{slug}: 引用的脚手架模块不存在: {path.name}"


@pytest.mark.parametrize("slug,step_slug", STEP_CASES)
def test_step_meta_and_five_piece_files(slug, step_slug):
    """(4) meta 不得含 order，tags 首项 project，字段与台账一致，五件套齐全。"""
    ledger = _ledger(slug)
    row = next(s for s in ledger["steps"] if s["slug"] == step_slug)
    directory = _step_dir(PROJECTS_DIR / slug, row)
    missing = [str(directory / f) for f in ROOT_FILES if not (directory / f).is_file()]
    missing += [
        str(directory / "variants" / "pytorch" / f)
        for f in VARIANT_FILES
        if not (directory / "variants" / "pytorch" / f).is_file()
    ]
    assert not missing, f"{slug}/{row['slug']}: 五件套缺失 {missing}"

    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    assert "order" not in meta, f"{slug}/{row['slug']}: 项目步骤不得写 order（不占全库编号池）"
    assert meta.get("tags", [])[:1] == ["project"], "tags 首项必须为 project"
    assert str(meta.get("title_en", "")).strip(), "缺少 title_en"
    assert meta.get("title") == row.get("title")
    assert meta.get("project") == slug
    assert meta.get("step") == row["step"]
    assert meta.get("depends_on") == row.get("depends_on")
    assert meta.get("verification") == row.get("verification")
    assert meta.get("scaffold_imports") == row.get("scaffold_imports")
    assert bool(meta.get("final_step")) == bool(row.get("final_step"))
    variants = meta.get("variants")
    assert isinstance(variants, dict) and "pytorch" in variants, "判题机依赖 variants 映射"


@pytest.mark.parametrize("slug", PROJECT_SLUGS)
def test_architecture_and_overview_are_bilingual(slug):
    """(5) 开篇页/总览页存在且双语齐备：heading 层级与代码块数量一致、英文无中文残留。"""
    ledger = _ledger(slug)
    project_dir = PROJECTS_DIR / slug
    for key, default in (("architecture_page", "architecture.md"), ("overview_page", "overview.md")):
        name = ledger.get(key, default)
        stem = name[:-3] if name.endswith(".md") else name
        zh = project_dir / f"{stem}.md"
        en = project_dir / f"{stem}.en.md"
        assert zh.is_file(), f"{slug}: 缺 {zh.name}"
        assert en.is_file(), f"{slug}: 缺 {en.name}"
        ztext, etext = zh.read_text(encoding="utf-8"), en.read_text(encoding="utf-8")
        assert _headings(ztext) == _headings(etext), f"{slug}/{stem}: 双语标题层级不一致"
        assert _fence_count(ztext) == _fence_count(etext), f"{slug}/{stem}: 双语代码块数量不一致"
        assert not HAN.search(etext), f"{slug}/{en.name}: 英文页仍有中文残留"


@pytest.mark.parametrize("slug", PROJECT_SLUGS)
def test_scaffold_modules_have_no_hidden_state(slug):
    """(2 纪律) 脚手架禁隐藏状态/随机/时钟/网络，否则状态会沿链泄漏。"""
    scaffold_dir = PROJECTS_DIR / slug / "scaffold"
    files = sorted(scaffold_dir.glob("*.py"))
    assert files, f"{slug}: scaffold/ 为空"
    assert (scaffold_dir / "__init__.py").is_file(), "scaffold/ 需要 __init__.py"
    # 按「代码行」判定，避免把 docstring 里说明禁则的英文句子当成违规代码
    banned = {
        "随机/时钟/网络/IO 导入": re.compile(
            r"^\s*(?:import|from)\s+(?:random|time|requests|urllib|socket|os)\b", re.M
        ),
        "global 语句": re.compile(r"^\s*global\s+\w", re.M),
        "sleep": re.compile(r"\btime\.sleep\("),
        "文件 IO": re.compile(r"(?<!\.)\bopen\("),
    }
    for path in files:
        text = path.read_text(encoding="utf-8")
        for label, pattern in banned.items():
            assert not pattern.search(text), (
                f"{slug}/scaffold/{path.name}: 含禁用项「{label}」"
            )


def test_projects_are_not_registered_in_normal_collections():
    """(5 隔离) 项目步骤不进 collections.json，也不与普通题 slug 撞名。"""
    collections = json.loads(COLLECTIONS_FILE.read_text(encoding="utf-8"))
    registered = {s for c in collections for s in c.get("problems", [])}
    normal_slugs = {
        d.name for d in (ROOT / "problems" / "pytorch").iterdir() if d.is_dir()
    }
    for slug in PROJECT_SLUGS:
        ledger = _ledger(slug)
        for row in ledger["steps"]:
            assert row["slug"] not in registered, f"{row['slug']}: 项目步骤混进了普通合集"
            assert row["slug"] not in normal_slugs, f"{row['slug']}: 与普通题 slug 撞名"
        assert slug not in registered, f"{slug}: 项目 slug 混进了普通合集"
