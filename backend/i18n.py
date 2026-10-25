"""User-facing zh/en messages shared by the API and judge engine."""

from __future__ import annotations

MESSAGES: dict[str, dict[str, str]] = {
    "problem_not_found": {"zh": "题目不存在", "en": "Problem not found"},
    "collection_not_found": {"zh": "合集不存在", "en": "Collection not found"},
    "solution_locked": {
        "zh": "通过全部测试用例后可查看参考答案",
        "en": "Pass all test cases to view the reference solution",
    },
    "import_json_invalid": {
        "zh": "无法解析 JSON，请选择 leet-infra 导出的进度文件",
        "en": "Cannot parse JSON. Choose a progress file exported by leet-infra.",
    },
    "import_format_invalid": {
        "zh": "进度文件格式不正确（缺少 solved 字段）",
        "en": "Invalid progress file: missing the solved field.",
    },
    "judge_setup_hint": {
        "zh": "评测环境未找到（EVAL_PYTHON 指向的解释器不存在）。请先运行 scripts/setup_eval_env.sh 创建 .venv-eval。",
        "en": "Judge environment not found (the interpreter pointed to by EVAL_PYTHON is missing). Run scripts/setup_eval_env.sh to create .venv-eval.",
    },
    "judge_spawn_failed": {
        "zh": "无法启动评测进程（{python}）：{error}",
        "en": "Could not start the judge process ({python}): {error}",
    },
    "judge_schedule_timeout": {
        "zh": "评测调度超时",
        "en": "Judge scheduling timed out.",
    },
    "judge_cancelled": {
        "zh": "运行已取消，未记录任何进度。",
        "en": "Run cancelled. No progress was recorded.",
    },
    "judge_timeout": {
        "zh": "运行超时（>{seconds}s）。检查是否存在死循环或过大的计算规模。",
        "en": "Time limit exceeded (>{seconds}s). Check for infinite loops or excessive computation.",
    },
    "judge_result_corrupt": {
        "zh": "评测结果解析失败（worker 输出损坏）。",
        "en": "Could not parse the judge result (corrupted worker output).",
    },
    "judge_env_load_failed": {
        "zh": "评测环境加载失败（torch 导入异常）。\n{detail}",
        "en": "Judge environment failed to load (torch import error).\n{detail}",
    },
    "judge_worker_crashed": {
        "zh": "评测进程异常退出，已回退保护。\n{detail}",
        "en": "The judge process exited unexpectedly; the run was aborted safely.\n{detail}",
    },
}


def normalize(lang: str | None) -> str:
    """Only zh/en are supported; anything else falls back to Chinese."""
    return "en" if lang == "en" else "zh"


def tr(lang: str | None, key: str, **params) -> str:
    text = MESSAGES[key][normalize(lang)]
    return text.format(**params) if params else text
