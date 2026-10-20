"""API 集成测试：题库列表 / 详情 / 运行（样例） / 提交（全量） / 答案解锁 / 进度。"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend import judge, progress  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "DATA_FILE", tmp_path / "progress.json")
    problems_mod = sys.modules["backend.problems"]
    monkeypatch.setattr(problems_mod, "_INDEX", None)
    from backend.app import app

    with TestClient(app) as c:
        yield c


def test_list_problems_returns_five_pytorch(client):
    data = client.get("/api/problems").json()
    assert len(data) == 5
    assert {p["framework"] for p in data} == {"pytorch"}
    slugs = {p["slug"] for p in data}
    assert "numerically-stable-softmax" in slugs
    assert all(p["solved"] is False for p in data)
    assert all(p["difficulty"] in ("Easy", "Medium", "Hard") for p in data)


def test_get_problem_detail(client):
    data = client.get("/api/problems/numerically-stable-softmax").json()
    assert data["title"] == "数值稳定 Softmax"
    assert "def solve(" in data["starter_code"]
    assert "softmax" in data["description"]


def test_get_missing_problem_404(client):
    assert client.get("/api/problems/does-not-exist").status_code == 404


def test_solution_locked_until_accepted(client):
    resp = client.get("/api/problems/numerically-stable-softmax/solution")
    assert resp.status_code == 403

    problem = sys.modules["backend.problems"].get_problem("numerically-stable-softmax")
    if not Path(judge.EVAL_PYTHON).exists():
        pytest.skip("评测环境未安装")
    sub = client.post(
        "/api/problems/numerically-stable-softmax/submit",
        json={"code": problem.solution()},
    ).json()
    assert sub["accepted"] is True
    assert sub["mode"] == "submit"

    resp = client.get("/api/problems/numerically-stable-softmax/solution")
    assert resp.status_code == 200
    assert "def solve(" in resp.json()["solution_code"]

    progress_data = client.get("/api/progress").json()
    assert progress_data["solved"] == 1
    assert progress_data["total"] == 5


def test_run_executes_examples_only_and_marks_no_progress(client):
    if not Path(judge.EVAL_PYTHON).exists():
        pytest.skip("评测环境未安装")
    problem = sys.modules["backend.problems"].get_problem("numerically-stable-softmax")
    run = client.post(
        "/api/problems/numerically-stable-softmax/run",
        json={"code": problem.solution()},
    ).json()
    # 参考解在运行模式下也只会跑 2 个样例用例，且不记录进度、不解锁答案
    assert run["mode"] == "run"
    assert run["accepted"] is True
    assert run["total"] == 2
    assert client.get("/api/progress").json()["solved"] == 0
    assert client.get("/api/problems/numerically-stable-softmax/solution").status_code == 403


def test_submit_wrong_answer_does_not_mark_solved(client):
    if not Path(judge.EVAL_PYTHON).exists():
        pytest.skip("评测环境未安装")
    sub = client.post(
        "/api/problems/numerically-stable-softmax/submit",
        json={"code": "def solve(logits, mask=None, dim=-1):\n    return logits\n"},
    ).json()
    assert sub["accepted"] is False
    assert sub["passed"] < sub["total"]
    assert client.get("/api/progress").json()["solved"] == 0
