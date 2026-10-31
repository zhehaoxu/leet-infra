"""题库完整性护栏：纯读文件系统、零运行时依赖（不导入 backend / torch / judge）。

兜住「靠人工记忆」最易漏的四类结构漂移，加/删/改题目后立即暴露：
  (a) collections.json 引用的每个 slug 都在盘上有对应题目录（删题/改 slug 后不留悬挂引用）。
  (b) 每题五件套齐全：meta.json / description.md / notes.md + variants/pytorch/{starter,solution,test_cases}.py。
  (c) 每个含 `operator` 标签的题都至少有一个领域标签能在前端 problemFilter.ts 的 TAG_FAMILY 命中族
      —— 否则前端 familyOf 会把它静默归入「框架 / 其他」，新族漏登记不报错也不红。
      （框架题本身不带 operator，合法落入「框架 / 其他」，故不参与 (c) 断言。）
  (d) 每题英文三件套齐全：title_en + description.en.md + notes.en.md，合集也有 name_en
      —— 前端切到 EN 时后端会回落中文，漏译只能靠这条护栏发现。
  (e) 每题 meta.json 的 order 是正整数且**全库唯一**（order 是全局编号池，脚手架按 max+1 分配；
      并行出题或手工补挂最易重复占号，重复不会报错、只会让题号显示歧义）。

命名以 test_ 前缀被后端 pytest 采集；因不依赖评测环境，`--smoke` 与 `-n auto` 均可叠加、秒级完成。
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROBLEMS_DIR = ROOT / "problems"
PYTORCH_DIR = PROBLEMS_DIR / "pytorch"
COLLECTIONS_FILE = PROBLEMS_DIR / "collections.json"
FILTER_TS = ROOT / "frontend" / "src" / "utils" / "problemFilter.ts"

# 每题根目录必备文件 + variants/pytorch 下必备三件套
ROOT_FILES = ("meta.json", "description.md", "notes.md")
VARIANT_FILES = ("starter.py", "solution.py", "test_cases.py")
# 英文版必备：两份 Markdown + meta.json 里的 title_en
ROOT_EN_FILES = ("description.en.md", "notes.en.md")


def _tag_family_keys() -> set[str]:
    """从前端 problemFilter.ts 抽取 TAG_FAMILY 的键集合（对象体内无嵌套花括号，直接匹配）。"""
    text = FILTER_TS.read_text(encoding="utf-8")
    block = re.search(r"TAG_FAMILY[^{]*\{(.*?)\}", text, re.S)
    assert block, "problemFilter.ts 中未找到 TAG_FAMILY 定义，护栏无法校验领域标签"
    return set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*:", block.group(1)))


TAG_FAMILY_KEYS = _tag_family_keys()


def _problem_dirs() -> list[str]:
    """盘上全部题目录名（跳过 __pycache__ 等下划线/点开头目录）。"""
    return sorted(
        d.name for d in PYTORCH_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(("_", "."))
    )


PROBLEM_DIRS = _problem_dirs()


def test_problem_bank_is_not_empty():
    """护栏前置：题库扫描确非空，避免 glob 失配让其余用例假绿。"""
    assert PROBLEM_DIRS, f"未在 {PYTORCH_DIR} 扫到任何题目目录"
    assert TAG_FAMILY_KEYS, "TAG_FAMILY 键集合为空"


# ---- (a) collections.json 引用不悬挂 ------------------------------------------------
def test_collection_slugs_all_exist_on_disk():
    collections = json.loads(COLLECTIONS_FILE.read_text(encoding="utf-8"))
    dangling = [
        f"{c['slug']} -> {slug}"
        for c in collections
        for slug in c.get("problems", [])
        if not (PYTORCH_DIR / slug).is_dir()
    ]
    assert not dangling, "collections.json 引用了盘上不存在的题目录（悬挂引用）:\n  " + "\n  ".join(dangling)


# ---- (b) 每题五件套齐全 -------------------------------------------------------------
@pytest.mark.parametrize("slug", PROBLEM_DIRS)
def test_problem_has_five_piece_files(slug):
    d = PYTORCH_DIR / slug
    missing = [str(d / f) for f in ROOT_FILES if not (d / f).is_file()]
    missing += [str(d / "variants" / "pytorch" / f) for f in VARIANT_FILES
                if not (d / "variants" / "pytorch" / f).is_file()]
    assert not missing, f"{slug} 缺少题库文件: {missing}"


# ---- (c) operator 题的领域标签命中前端族 -------------------------------------------
@pytest.mark.parametrize("slug", PROBLEM_DIRS)
def test_operator_problem_maps_to_a_family(slug):
    meta = json.loads((PYTORCH_DIR / slug / "meta.json").read_text(encoding="utf-8"))
    tags = set(meta.get("tags", []))
    if "operator" not in tags:
        pytest.skip("框架题不带 operator 标签，合法落入「框架 / 其他」，不校验领域标签")
    hit = tags & TAG_FAMILY_KEYS
    assert hit, (
        f"{slug} 标了 operator 却没有任何标签命中前端 TAG_FAMILY "
        f"（tags={sorted(tags)}）——会被 familyOf 静默归入「框架 / 其他」；"
        f"请在 problemFilter.ts 的 TAG_FAMILY 补登记该领域标签"
    )


# ---- (d) 双语内容齐全 ---------------------------------------------------------------
@pytest.mark.parametrize("slug", PROBLEM_DIRS)
def test_problem_has_english_pair(slug):
    d = PYTORCH_DIR / slug
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    missing = [str(d / f) for f in ROOT_EN_FILES if not (d / f).is_file()]
    if not meta.get("title_en", "").strip():
        missing.append(f"{slug}/meta.json: title_en")
    assert not missing, f"{slug} 缺少英文内容，EN 语言下会回落中文: {missing}"


def test_collections_have_english_names():
    collections = json.loads(COLLECTIONS_FILE.read_text(encoding="utf-8"))
    missing = [c["slug"] for c in collections if not c.get("name_en", "").strip()]
    assert not missing, f"合集缺少 name_en，EN 语言下合集名会回落中文: {missing}"


# ---- (e) order 全库唯一 -------------------------------------------------------------
def test_problem_order_is_globally_unique():
    """order 是跨合集的全局编号池（add_problem.py 取 max+1），重复占号必须显式失败。

    后端 problems.all_problems() 以 (framework, order, slug) 排序，重复号不会报错、
    只会静默退化成按 slug 排，所以题号歧义只能靠这条护栏拦住。
    """
    seen: dict[int, list[str]] = {}
    bad_type = []
    for slug in PROBLEM_DIRS:
        meta = json.loads((PYTORCH_DIR / slug / "meta.json").read_text(encoding="utf-8"))
        order = meta.get("order")
        if not isinstance(order, int) or isinstance(order, bool) or order <= 0:
            bad_type.append(f"{slug}: order={order!r}")
            continue
        seen.setdefault(order, []).append(slug)
    dups = {o: sorted(s) for o, s in seen.items() if len(s) > 1}
    assert not bad_type, f"order 必须为正整数（未填/非整数会落到排序首位）: {sorted(bad_type)}"
    assert not dups, (
        "以下 order 被多题重复占用，新题请扫描全库取未占用号段："
        + "; ".join(f"{o} -> {', '.join(v)}" for o, v in sorted(dups.items()))
    )
