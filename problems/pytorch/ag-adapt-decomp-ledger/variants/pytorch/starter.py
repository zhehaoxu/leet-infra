"""ADaPT 按需递归分解账。

实现 `solve(task, exec_table, plan_table, ...)`：把官方 `plan_and_run` 的递归控制流
算成一棵执行账。语义要求见 description.md，这里只留签名与骨架。
"""

from __future__ import annotations


def solve(task, exec_table, plan_table, max_depth=3, max_runs=20, max_patience=5,
          info_prop_mode="last-step-last-act"):
    """跑一次 ADaPT 分解，返回 `{"nodes": [...], "runs": int, ...}`。

    参数
    ----
    task : str
        根任务文本。
    exec_table : dict
        `{任务文本: {"steps": [{"act", "obs", "reward", "done"}, ...]}}`：执行器脚本，
        脚本跑完仍未 `done` 即算失败。缺条目应抛 `ValueError`。
    plan_table : dict
        `{任务文本: 规划器输出文本}`；需要分解却缺条目应抛 `ValueError`。
    max_depth : int
        深度上限，根深度为 1；到上限的节点**仍然先直接执行一次**。
    max_runs : int
        单次执行的动作上限，官方按 `range(1, max_runs)` 取，所以最多 `max_runs - 1` 步。
    max_patience : int
        连续「没进展」观测（`"Nothing happens."` / `"OK."`）达到该值即中断。
    info_prop_mode : str
        `"last-step-last-act"` 或 `"all-step-last-act"`。

    返回
    ----
    dict
        `nodes` / `runs` / `checkpoint` / `completion` / `info_prop` /
        `success` / `reward` / `terminate` / `depth`。
    """
    raise NotImplementedError
