"""全量自测：每道题的参考解必须能通过自己的全部用例（真实 judge 评测）。

兜底题面/用例数值错误——新题或改用例后必跑。

提速两档（可叠加）：
- 并行：加 pytest-xdist 的 `-n auto`，每个 worker 各自起 warm judge 进程，真并行。
- 冒烟：加 `--smoke`，只打 `/run` 跑每题样例用例（命中题面/示例数值的最快内循环），
  默认不带则打 `/submit` 跑全量用例。
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend import benchmarks, judge, problems, progress  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def _all_slugs():
    return [p.slug for p in problems.load_problems()]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "DATA_FILE", tmp_path / "progress.json")
    monkeypatch.setattr(benchmarks, "DATA_FILE", tmp_path / "benchmarks.json")
    # 让「已有基准」恒为真，抑制后台基准线程，保证 judge 串行
    monkeypatch.setattr(benchmarks, "get_benchmarks", lambda slug, variant: {"_": 0.0})
    monkeypatch.setattr(problems, "_INDEX", None)
    from backend.app import app

    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("slug", _all_slugs())
def test_reference_solution_gets_accepted(client, slug, pytestconfig):
    if not Path(judge.EVAL_PYTHON).exists():
        pytest.skip("评测环境未安装")
    smoke = pytestconfig.getoption("--smoke")
    endpoint = "run" if smoke else "submit"
    problem = problems.get_problem(slug)
    result = client.post(
        f"/api/problems/{slug}/{endpoint}",
        json={"code": problem.solution(), "variant": problem.default_variant},
    ).json()
    failed = [c["name"] for c in result["cases"] if not c["passed"]]
    assert result["accepted"] is True, f"{slug} 参考解未 AC，失败用例: {failed}"
