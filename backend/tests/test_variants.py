"""多实现 variant 行为测试：声明、回退、提交携带、进度按 slug、参考解按 variant。"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend import judge, problems, progress  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "DATA_FILE", tmp_path / "progress.json")
    monkeypatch.setattr(problems, "_INDEX", None)
    from backend.app import app

    with TestClient(app) as c:
        yield c


def _require_eval_env():
    if not Path(judge.EVAL_PYTHON).exists():
        pytest.skip("评测环境未安装")


def test_summary_declares_variants(client):
    items = client.get("/api/problems").json()
    relu = next(p for p in items if p["slug"] == "relu")
    variants = {v["key"]: v for v in relu["variants"]}
    assert set(variants) == {"pytorch", "triton", "cuda"}
    assert variants["pytorch"]["available"] is True
    assert variants["triton"]["available"] is False
    assert variants["cuda"]["available"] is False
    # 框架题未声明 variants，默认单 pytorch
    tensor = next(p for p in items if p["slug"] == "tensor-shape-indexing")
    assert [v["key"] for v in tensor["variants"]] == ["pytorch"]


def test_detail_falls_back_unavailable_variant(client):
    data = client.get("/api/problems/relu?variant=triton").json()
    assert data["active_variant"] == "pytorch"  # 不可用回退默认，不报错
    assert "def solve(" in data["starter_code"]

    data = client.get("/api/problems/relu?variant=pytorch").json()
    assert data["active_variant"] == "pytorch"

    data = client.get("/api/problems/relu").json()
    assert data["active_variant"] == "pytorch"


def test_run_submit_carry_variant(client):
    _require_eval_env()
    problem = problems.get_problem("relu")
    r = client.post("/api/problems/relu/run", json={"code": problem.solution(), "variant": "cuda"})
    assert r.status_code == 200
    # 不可用 variant 回退 pytorch 评测，结果回显实际使用的 variant
    assert r.json()["variant"] == "pytorch"
    assert r.json()["accepted"] is True

    s = client.post("/api/problems/relu/submit", json={"code": problem.solution(), "variant": "pytorch"})
    assert s.json()["variant"] == "pytorch"
    assert s.json()["accepted"] is True


def test_progress_keyed_by_slug_not_variant(client):
    _require_eval_env()
    problem = problems.get_problem("softmax")
    client.post("/api/problems/softmax/submit", json={"code": problem.solution(), "variant": "triton"})
    # 任一（回退后的）variant 通过即按 slug 记进度
    assert client.get("/api/progress").json()["solved"] == 1
    # solution 解锁也按 slug，响应携带解析后的 variant
    resp = client.get("/api/problems/softmax/solution?variant=triton").json()
    assert resp["variant"] == "pytorch"
    assert "def solve(" in resp["solution_code"]
