"""项目里程碑参考解自测：每步必须带自己的官方脚手架全量 AC（project.md §6.4）。

与普通题库的 test_self_ac.py 分开：这里按 project.json 的 step 台账走
problems/projects/，并把 scaffold_imports 指定的冻结模块注入判题沙箱。
评测环境缺失时整体跳过；不写任何进度文件。
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend import judge, projects  # noqa: E402

PROJECTS_DIR = ROOT / "problems" / "projects"


def _cases():
    cases = []
    for entry in json.loads((PROJECTS_DIR / "projects.json").read_text(encoding="utf-8")):
        slug = entry["slug"]
        ledger = json.loads((PROJECTS_DIR / slug / "project.json").read_text(encoding="utf-8"))
        for row in ledger["steps"]:
            cases.append((slug, int(row["step"])))
    return cases


CASES = _cases()


def test_project_registry_has_steps():
    assert CASES, "项目注册表为空，自测会假绿"


@pytest.mark.parametrize("slug,step", CASES)
def test_project_reference_solution_gets_accepted(slug, step):
    if not Path(judge.EVAL_PYTHON).exists():
        pytest.skip("评测环境未安装")
    step_dir, _ledger, row, meta = projects.get_step(slug, step)
    project_dir = projects.get_project(slug)[0]
    vdir = projects.variant_dir(step_dir, None, meta)
    scaffold = projects.scaffold_sources(project_dir, row.get("scaffold_imports", []))
    result = judge.run_judgement(
        (vdir / "solution.py").read_text(encoding="utf-8"),
        (vdir / "test_cases.py").read_text(encoding="utf-8"),
        mode="submit",
        scaffold=scaffold,
    )
    failed = [c["name"] for c in result.get("cases", []) if not c["passed"]]
    assert result["accepted"], (
        f"{slug}/step {step} 未 AC：{result.get('status')} "
        f"{result.get('passed')}/{result.get('total')}；"
        f"失败用例={failed}；error={result.get('error')}"
    )
