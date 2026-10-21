"""Judge 单元测试：正确 / 错误 / 超时 / 语法错 四类提交的解析。"""

import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend import judge, problems  # noqa: E402

SAMPLE = problems.get_problem("numerically-stable-softmax")
assert SAMPLE is not None, "题库缺失：numerically-stable-softmax"
TESTS = SAMPLE.test_cases()


def _require_eval_env():
    if not Path(judge.EVAL_PYTHON).exists():
        pytest.skip("评测环境未安装（先运行 scripts/setup_eval_env.sh）")


def test_correct_submission_is_accepted():
    _require_eval_env()
    result = judge.run_judgement(SAMPLE.solution(), TESTS)
    assert result["status"] == "accepted", result
    assert result["accepted"] is True
    assert result["passed"] == result["total"] == 6
    assert all(case["passed"] for case in result["cases"])
    assert result["error"] is None


def test_run_mode_only_executes_example_cases():
    _require_eval_env()
    result = judge.run_judgement(SAMPLE.solution(), TESTS, mode="run")
    assert result["mode"] == "run"
    assert result["status"] == "accepted", result
    # softmax 题库标记了 2 个样例用例
    assert result["passed"] == result["total"] == 2
    names = {case["name"] for case in result["cases"]}
    assert names == {"basic_probability_simplex", "matches_torch_softmax_no_mask"}


def test_run_mode_falls_back_when_no_example_marks():
    _require_eval_env()
    plain_tests = '''
def check_a(solve):
    assert solve(1) == 2

def check_b(solve):
    assert solve(2) == 4
'''
    result = judge.run_judgement("def solve(x):\n    return x * 2\n", plain_tests, mode="run")
    assert result["total"] == 2  # 未标记样例时回退为全部用例
    assert result["accepted"] is True


def test_wrong_submission_reports_failed_cases():
    _require_eval_env()
    code = """
import torch

def solve(logits, mask=None, dim=-1):
    return torch.softmax(logits, dim=dim)  # 未处理 mask
"""
    result = judge.run_judgement(code, TESTS)
    assert result["status"] == "wrong_answer"
    assert result["accepted"] is False
    assert 0 < result["passed"] < result["total"]
    failed = [c for c in result["cases"] if not c["passed"]]
    assert failed and all(c["message"] for c in failed), "失败用例必须带错误消息"


def test_syntax_error_submission():
    _require_eval_env()
    result = judge.run_judgement("def solve(:\n  ???", TESTS)
    assert result["status"] == "syntax"
    assert result["accepted"] is False
    assert "SyntaxError" in (result["error"] or "")


def test_missing_solve_function():
    _require_eval_env()
    result = judge.run_judgement("x = 1\n", TESTS)
    assert result["status"] == "no_solve"
    assert "solve" in (result["error"] or "")


def test_timeout_submission(monkeypatch):
    _require_eval_env()
    monkeypatch.setattr(judge, "RUN_TIMEOUT_S", 5)
    code = """
import torch

def solve(logits, mask=None, dim=-1):
    while True:
        pass
"""
    result = judge.run_judgement(code, TESTS)
    assert result["status"] == "timeout"
    assert result["accepted"] is False


def test_runtime_exception_is_not_crash():
    _require_eval_env()
    code = """
import torch

def solve(logits, mask=None, dim=-1):
    raise RuntimeError("boom")
"""
    result = judge.run_judgement(code, TESTS)
    assert result["status"] == "wrong_answer"
    assert all(not c["passed"] for c in result["cases"])
    assert "boom" in result["cases"][0]["message"]


def test_user_stdout_is_captured():
    _require_eval_env()
    code = SAMPLE.solution() + '\nprint("hello from user")\n'
    result = judge.run_judgement(code, TESTS)
    assert "hello from user" in result["stdout"]
    assert judge.RESULT_MARKER not in result["stdout"]


def test_warm_worker_reuse_keeps_results_correct():
    """连续两次评测（第二次复用已预热进程）结果均正确。"""
    _require_eval_env()
    r1 = judge.run_judgement(SAMPLE.solution(), TESTS, mode="run")
    assert r1["status"] == "accepted", r1
    r2 = judge.run_judgement(SAMPLE.solution(), TESTS, mode="run")
    assert r2["status"] == "accepted", r2
    assert r2["total"] == 2


def test_cancel_running_stops_evaluation():
    """死循环评测发起后从另一线程取消，应返回 cancelled 而非等超时。"""
    _require_eval_env()
    loop_code = """
import torch

def solve(logits, mask=None, dim=-1):
    while True:
        pass
"""
    out = {}
    def run():
        out["r"] = judge.run_judgement(loop_code, TESTS)

    t = threading.Thread(target=run)
    t.start()
    cancelled = False
    deadline = time.time() + 40  # 等 worker 就绪并开始评测
    while time.time() < deadline and not cancelled:
        cancelled = judge.cancel_running()
        if not cancelled:
            time.sleep(0.2)
    t.join(30)
    assert cancelled, "未能触发取消"
    assert out["r"]["status"] == "cancelled", out["r"]
    assert out["r"]["accepted"] is False
