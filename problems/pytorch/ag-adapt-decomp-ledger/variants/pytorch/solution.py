"""ADaPT 按需递归分解账（arXiv 2311.05772，archiki/ADaPT main run_alfworld.py）。

把官方 `plan_and_run` / `alfworld_run` / `plan_to_args` 剥成纯函数：LLM 调用全变查表
（`exec_table` 给执行器脚本、`plan_table` 给规划器文本），环境步进变成脚本里的
`{"act", "obs", "reward", "done"}`，只保留「谁在什么深度先试一次、失败了怎么分解、
AND/OR 怎么短路、账怎么攒」。

核心口径（全部对齐官方实现）：
- **按需**：每个 str 任务先自己跑一次执行器；成功或到深度上限就返回，否则才规划。
  所以 `max_depth=1` 时**一次分解都不会发生**。
- 规划结果是一棵 `{"steps": [...], "logic": "AND"/"OR"}` 的嵌套结构；括号里的子式
  是**已解析好的子计划**，直接执行、不再试执行器、不再规划、也不加深度。
- `AND` 遇失败立刻返回；`OR` 遇成功立刻返回，两者的账各自不同。
- 只把**成功**子节点的 completion 拼进 running_completion；但到了深度上限而直接
  执行失败时，那次失败的 completion 也照样拼进去（官方 `running_completion +=
  completion` 在 `if succ` 之外）。
"""

from __future__ import annotations

import copy
import re

# 环境默认回应（官方 alfworld_run 里把这两种观测都算「没进展」）
_PATIENCE_OBS = ("Nothing happens.", "OK.")
_TOKEN_RE = re.compile(r"Step \d+|AND|OR|\(|\)")


def _execute(task, exec_table, max_runs, max_patience):
    """跑一次执行器脚本（官方 alfworld_run 的无环境版）。

    官方口径里的四个坑都保留：`range(1, max_runs)` 少跑一步、耐心值用 `==` 判定、
    触发耐心的那一步**不写进** completion、循环自然结束（非 done）时**丢弃** reward
    并把 success 压成 False。
    """
    if task not in exec_table:
        raise ValueError("no execution script for task %r" % (task,))
    spec = exec_table[task] or {}
    steps = spec.get("steps") or []

    history, parts = [], []
    success = terminate = done = False
    reward = 0
    patience = 0
    turns = 0

    for i in range(1, max_runs):
        if i - 1 >= len(steps):
            break
        step = steps[i - 1]
        if "act" not in step:
            raise ValueError("step %d of %r has no act" % (i, task))
        act = step["act"]
        obs = step.get("obs", "Nothing happens.")
        step_reward = int(step.get("reward", 0))
        done = bool(step.get("done", False))
        turns += 1

        if act.startswith("think:"):
            obs = "OK."
            low = act.lower()
            if "task completed!" in low:
                done = True
                success = True
            if "task failed!" in low:
                done = True
                success = False
        else:
            history.append(act)

        if obs in _PATIENCE_OBS:
            patience += 1
            if patience == max_patience:
                terminate = True
                break
        else:
            patience = 0

        parts.append(" %s\n%s\n>" % (act, obs))
        if step_reward:
            reward = step_reward
            success = True
            terminate = False
        if done:
            break

    if not done:
        success = False
        terminate = True
        reward = 0

    return {
        "success": success,
        "reward": reward,
        "terminate": terminate,
        "turns": turns,
        "history": history,
        "completion": "".join(parts),
    }


def _parse_expression(expression):
    """把执行顺序表达式解析成嵌套 dict（官方 parse_expression 的直译）。

    词法器只认 `Step N` / `AND` / `OR` / 括号，别的一律忽略；同一层出现多个逻辑词时
    **后写的覆盖先写的**（`A AND B OR C` → 整层 OR）。
    """
    stack, current = [], {}
    for token in _TOKEN_RE.findall(expression or ""):
        if token.startswith("Step"):
            current.setdefault("steps", []).append(int(token.split()[1]))
        elif token in ("AND", "OR"):
            current["logic"] = token
        elif token == "(":
            stack.append(current)
            current = {}
        elif token == ")":
            closed = current
            current = stack.pop()
            current.setdefault("steps", []).append(closed)
    return current


def _fetch_args(lookup, expression):
    """把叶子里的 Step 编号换成子任务文本（官方 fetch_args 的直译）。"""
    out = copy.deepcopy(expression)
    assert "steps" in expression
    for index, step in enumerate(expression["steps"]):
        if isinstance(step, int):
            out["steps"][index] = lookup[step]
        elif isinstance(step, dict):
            out["steps"][index] = _fetch_args(lookup, step)
    return out


def _plan_to_args(plan, keyword="Step", order_key="execution order"):
    """规划文本 → 嵌套子任务树（官方 plan_to_args 的直译）。

    解析失败（没有执行顺序行 / 括号不配对 / Step 编号越界）时**吞掉整个结构**，
    退化成 `{"steps": [原文...], "logic": "AND"}` —— 括号里的嵌套就此丢失。
    """
    args, logic = [], None
    for line in (plan or "").split("\n"):
        if line.startswith(keyword):
            args.append(re.sub(r"{} \d+: ".format(keyword), "", line))
        if order_key in line.lower():
            logic = line.split(": ")[-1]
    lookup = {i + 1: args[i] for i in range(len(args))}
    try:
        return _fetch_args(lookup, _parse_expression(logic))
    except Exception:  # noqa: BLE001 —— 官方就是裸 except
        return {"steps": args, "logic": "AND"}


def _salient_info(completion):
    """从 completion 里抠出「最后一次非 think 的真实动作 + 观测」。

    官方口径：必须字面含 `Task completed!`（**大小写敏感**，而执行器判定成功用的是
    小写包含，两者不一致）；先砍掉末尾两行，再跳过尾部的 think 对；第二段及以后的
    act 行带着上一行残留的 `> ` 前缀。
    """
    completion = completion.rstrip("\n>")
    if "Task completed!" not in completion:
        return ""
    lines = completion.split("\n")[:-2]
    while len(lines) >= 2 and "think:" in lines[-2]:
        lines = lines[:-2]
    if len(lines) < 2:
        return ""
    act, obs = lines[-2], lines[-1]
    return "\n".join([act.replace("Task completed!", ""), obs])


class _Runner:
    """一次完整分解的账本状态：节点表、轮次计数、checkpoint、info 串联。"""

    def __init__(self, exec_table, plan_table, max_depth, max_runs,
                 max_patience, mode):
        self.exec_table = exec_table
        self.plan_table = plan_table
        self.max_depth = max_depth
        self.max_runs = max_runs
        self.max_patience = max_patience
        self.mode = mode
        self.nodes = []
        self.runs = 0
        self.checkpoint = []

    # -- info 传播：两种模式在子节点成功路径上不对称（all-step 那条根本不更新）--
    def _update_info(self, past_info, running):
        if self.mode == "last-step-last-act":
            return _salient_info(running)
        return (past_info + "\n" + _salient_info(running)).lstrip("\n")

    def run(self, task, depth, past_info):
        record = {
            "id": len(self.nodes),
            "task": copy.deepcopy(task),
            "depth": depth,
            "replayed": len(self.checkpoint),
            "children": [],
            "direct": None,
            "plan": None,
            "stop": None,
            "success": False,
            "reward": 0,
        }
        self.nodes.append(record)

        if isinstance(task, str):
            attempt = _execute(task, self.exec_table, self.max_runs,
                               self.max_patience)
            self.runs += attempt["turns"]
            record["direct"] = {
                "success": attempt["success"],
                "reward": attempt["reward"],
                "terminate": attempt["terminate"],
                "turns": attempt["turns"],
                "history": attempt["history"],
            }
            running = ""
            if attempt["success"] or depth >= self.max_depth:
                if attempt["success"]:
                    self.checkpoint.extend(attempt["history"])
                running += attempt["completion"] + "\n"
                info = past_info
                if attempt["success"]:
                    info = self._update_info(past_info, running)
                    record["stop"] = "success"
                else:
                    record["stop"] = "depth-cap"
                record["success"] = attempt["success"]
                record["reward"] = attempt["reward"]
                return {
                    "success": attempt["success"],
                    "reward": attempt["reward"],
                    "terminate": attempt["terminate"],
                    "completion": running,
                    "depth": depth,
                    "info": info,
                }

            if task not in self.plan_table:
                raise ValueError("no plan for task %r" % (task,))
            parsed = _plan_to_args(self.plan_table[task])
            self.runs += 1
            record["plan"] = copy.deepcopy(parsed)
            if len(parsed["steps"]) == 1:
                only = parsed["steps"][0]
                if isinstance(only, str):
                    parsed = {"steps": [only], "logic": "AND"}
                elif isinstance(only, dict):
                    parsed = only
                    parsed.setdefault("logic", "AND")
            depth += 1
        else:
            # 括号子式：已经是解析好的计划，直接执行，不试执行器、不规划、不加深度
            parsed = copy.deepcopy(task)

        if not isinstance(parsed.get("steps"), list):
            raise ValueError("plan has no step list: %r" % (parsed,))
        if not parsed["steps"]:
            raise ValueError("plan produced no steps")
        parsed.setdefault("logic", "AND")
        record["plan"] = copy.deepcopy(parsed)
        record["child_depth"] = depth
        logic = str(parsed["logic"]).lower()

        running = ""
        info = past_info
        last = None
        for sub in parsed["steps"]:
            child_id = len(self.nodes)
            child = self.run(sub, depth, info)
            record["children"].append(child_id)
            last = child
            if logic == "or" and child["success"]:
                running += child["completion"] + "\n"
                if self.mode == "last-step-last-act":
                    info = _salient_info(running)
                record["stop"] = "or-success"
                record["success"] = True
                record["reward"] = child["reward"]
                return {
                    "success": True,
                    "reward": child["reward"],
                    "terminate": child["terminate"],
                    "completion": running,
                    "depth": depth,
                    "info": info,
                }
            if child["success"]:
                running += child["completion"] + "\n"
                if self.mode == "last-step-last-act":
                    info = _salient_info(running)
            if logic == "and" and not child["success"]:
                record["stop"] = "and-fail"
                record["reward"] = child["reward"]
                return {
                    "success": False,
                    "reward": child["reward"],
                    "terminate": child["terminate"],
                    "completion": running,
                    "depth": depth,
                    "info": info,
                }

        record["stop"] = "children-done"
        record["success"] = last["success"]
        record["reward"] = last["reward"]
        return {
            "success": last["success"],
            "reward": last["reward"],
            "terminate": last["terminate"],
            "completion": running,
            "depth": depth,
            "info": info,
        }


def solve(task, exec_table, plan_table, max_depth=3, max_runs=20, max_patience=5,
          info_prop_mode="last-step-last-act"):
    """跑一次 ADaPT 分解，返回整棵执行账。

    参数
    ----
    task : str
        根任务。
    exec_table : dict
        `{任务文本: {"steps": [{"act", "obs", "reward", "done"}, ...]}}`，执行器脚本；
        脚本跑完还没 done 就算失败（官方口径）。缺条目 → ValueError。
    plan_table : dict
        `{任务文本: 规划器输出文本}`。需要分解时缺条目 → ValueError。
    max_depth : int
        深度上限，根为 1；到上限的节点仍然会先直接执行一次。
    max_runs : int
        单次执行的动作上限，官方 `range(1, max_runs)` 所以实际最多 `max_runs - 1` 步。
    max_patience : int
        连续「没进展」观测达到该值就中断（官方用 `==` 判定，只看这一条路径）。
    info_prop_mode : str
        `"last-step-last-act"` 或 `"all-step-last-act"`，两者在子节点成功路径上不对称。

    返回
    ----
    dict
        `nodes`（先序节点账）、`success`/`reward`/`terminate`、`runs`（动作数 + 规划数）、
        `checkpoint`（成功的真实动作序列）、`completion`、`info_prop`、`depth`。
    """
    if info_prop_mode not in ("last-step-last-act", "all-step-last-act"):
        raise ValueError("unknown info_prop_mode: %r" % (info_prop_mode,))
    if max_depth < 1:
        raise ValueError("max_depth must be >= 1")
    if max_patience < 1:
        raise ValueError("max_patience must be >= 1")
    if not isinstance(task, str) or not task:
        raise ValueError("task must be a non-empty string")

    runner = _Runner(exec_table, plan_table, max_depth, max_runs, max_patience,
                     info_prop_mode)
    out = runner.run(task, 1, "")
    return {
        "nodes": runner.nodes,
        "runs": runner.runs,
        "checkpoint": runner.checkpoint,
        "completion": out["completion"],
        "info_prop": out["info"],
        "success": out["success"],
        "reward": out["reward"],
        "terminate": out["terminate"],
        "depth": out["depth"],
    }
