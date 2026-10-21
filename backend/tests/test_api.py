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


def test_collections_list_and_detail(client):
    data = client.get("/api/collections").json()
    assert {c["slug"] for c in data} == {"pytorch", "operators"}
    pytorch = next(c for c in data if c["slug"] == "pytorch")
    assert pytorch["problem_count"] == 5
    assert pytorch["solved_count"] == 0
    operators = next(c for c in data if c["slug"] == "operators")
    assert operators["problem_count"] == 3

    detail = client.get("/api/collections/operators").json()
    slugs = [p["slug"] for p in detail["problems"]]
    assert slugs == [
        "numerically-stable-softmax",
        "layernorm-forward",
        "multi-head-attention-forward",
    ]
    assert client.get("/api/collections/nope").status_code == 404


def test_problem_summary_carries_collection_refs(client):
    items = client.get("/api/problems").json()
    softmax = next(p for p in items if p["slug"] == "numerically-stable-softmax")
    refs = {c["slug"] for c in softmax["collections"]}
    assert refs == {"pytorch", "operators"}  # 同时属于两个合集
    # 合集详情页的题目也带合集字段
    detail = client.get("/api/collections/operators").json()
    assert detail["problems"][0]["collections"]


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


def test_progress_reset_clears_solved_and_relocks_solution(client):
    if not Path(judge.EVAL_PYTHON).exists():
        pytest.skip("评测环境未安装")
    problem = sys.modules["backend.problems"].get_problem("numerically-stable-softmax")
    sub = client.post(
        "/api/problems/numerically-stable-softmax/submit",
        json={"code": problem.solution()},
    ).json()
    assert sub["accepted"] is True
    assert client.get("/api/problems/numerically-stable-softmax/solution").status_code == 200

    after = client.post("/api/progress/reset").json()
    assert after["solved"] == 0
    assert after["total"] == 5
    # 重置后参考答案重新锁定
    assert client.get("/api/problems/numerically-stable-softmax/solution").status_code == 403


def test_judge_cancel_endpoint_no_active_task(client):
    resp = client.post("/api/judge/cancel")
    assert resp.status_code == 200
    assert resp.json() == {"cancelled": False}
