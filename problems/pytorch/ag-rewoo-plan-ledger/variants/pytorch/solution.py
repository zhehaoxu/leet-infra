# ReWOO（Xu et al., 2305.18323）官方 billxbf/ReWOO 的 algos/PWS.py 标量化：
# Planner 输出（plan 文本）作显式入参，Worker 输出按证据号显式入参，
# Solver 输出显式入参——只考 plan 解析、#E 占位符替换、worker_log 装配
# 与 token/成本账。
# 官方不建模（记 notes）：真实 LLM/Google/Calculator 调用、Calculator 的
# langchain LLMMathChain prompt 模板常量（故 Calculator 计数不入题）、
# planner/solver 的 prompt 模板与 fewshot、wall_time。


def _model_unit_price(model):
    """官方 get_token_unit_price 价格表（$/token）。"""
    if model == "text-davinci-003":
        return 0.00002
    if model == "gpt-3.5-turbo":
        return 0.000002
    if model == "gpt-4":
        return 0.00003
    if model in ("tloen/alpaca-lora-7b", "rewoo/planner_7B"):
        return 0.0
    raise ValueError("Model not found")


def solve(planner_response, worker_outputs, solver_output,
          planner_tokens=0, solver_tokens=0, tools=("Google", "LLM"),
          planner_model="text-davinci-003", solver_model="text-davinci-003"):
    """PWS（Plan-Worker-Solve）账。

    - planner_response: Planner 的原始输出文本（多行）；
      以 "Plan:" 开头的行记入 plans；"#E<d> = ..." 行解析为证据声明，
      **仅 len("#Ed") == 3（单个数字下标）有效**，其余记 "No evidence found"。
    - worker_outputs: {证据号: 工具输出}，供有效工具调用的证据使用。
    - solver_output: Solver 的最终答案（显式入参）。
    - planner_tokens/solver_tokens: 两次 LLM 调用的 prompt+completion 合计
      token 数（显式入参）。
    - tools: 可用工具名（官方 available_tools）。

    返回账本 dict（plans / planner_evidences / worker_calls /
    worker_evidences / worker_log / output / tool_usage / steps /
    total_tokens / token_cost / tool_cost / total_cost）。
    """
    if not isinstance(planner_response, str):
        raise ValueError("planner_response must be a str")
    if not isinstance(worker_outputs, dict):
        raise ValueError("worker_outputs must be a dict")
    if not isinstance(solver_output, str):
        raise ValueError("solver_output must be a str")
    if not isinstance(tools, (list, tuple)) or \
            not all(isinstance(t, str) for t in tools):
        raise ValueError("tools must be a list of tool names")
    if isinstance(planner_tokens, bool) or not isinstance(planner_tokens, int) \
            or planner_tokens < 0:
        raise ValueError("planner_tokens must be a non-negative int")
    if isinstance(solver_tokens, bool) or not isinstance(solver_tokens, int) \
            or solver_tokens < 0:
        raise ValueError("solver_tokens must be a non-negative int")
    planner_price = _model_unit_price(planner_model)
    solver_price = _model_unit_price(solver_model)

    # _parse_plans：以 "Plan:" 开头的行
    plans = [line for line in planner_response.splitlines()
             if line.startswith("Plan:")]

    # _parse_planner_evidences：'#' + 'E' + 数字开头，按第一个 '=' 切分；
    # 证据号 strip 后必须恰为 "#Ed"（len 3，单数字），否则记 No evidence found
    planner_evidences = {}
    for line in planner_response.splitlines():
        if line.startswith("#") and len(line) >= 3 \
                and line[1] == "E" and line[2].isdigit():
            if "=" not in line:
                raise ValueError(f"malformed evidence line: {line!r}")
            e, tool_call = line.split("=", 1)
            e, tool_call = e.strip(), tool_call.strip()
            if len(e) == 3:
                planner_evidences[e] = tool_call
            else:
                planner_evidences[e] = "No evidence found"

    # _get_worker_evidences：按 planner_evidences 插入序逐条执行
    worker_evidences = {}
    worker_calls = {}
    tool_counter = {}
    for e, tool_call in planner_evidences.items():
        if "[" not in tool_call:
            # 无工具调用的声明行：原文即证据（官方 pass-through）
            worker_evidences[e] = tool_call
            worker_calls[e] = None
            continue
        tool, tool_input = tool_call.split("[", 1)
        tool_input = tool_input[:-1]  # 官方盲删最后一个字符（约定为 ']'）
        # 占位符替换：按原文出现序，replace 全量替换（含插入文本里新出现的
        # 占位符）；尚未算出的前向引用原样保留
        for var in _find_vars(tool_input):
            if var in worker_evidences:
                tool_input = tool_input.replace(
                    var, "[" + worker_evidences[var] + "]")
        worker_calls[e] = tool_input
        if tool in tools:
            if e not in worker_outputs:
                raise ValueError(f"missing worker output for {e}")
            worker_evidences[e] = worker_outputs[e]
            if tool == "Google":
                tool_counter["Google"] = tool_counter.get("Google", 0) + 1
            elif tool == "LLM":
                tool_counter["LLM_token"] = tool_counter.get("LLM_token", 0) \
                    + len(tool_input + worker_evidences[e]) // 4
        else:
            worker_evidences[e] = "No evidence found"

    # worker_log：第 i 个 Plan 行与证据 "#E{i+1}" 按**下标**配对（官方
    # self.worker_evidences[e]，缺失即崩——本题改 ValueError，差异记 notes）
    worker_log = ""
    for i in range(len(plans)):
        e = f"#E{i + 1}"
        if e not in worker_evidences:
            raise ValueError(f"plan {i} has no matching evidence {e}")
        worker_log += f"{plans[i]}\nEvidence:\n{worker_evidences[e]}\n"

    llm_tokens = tool_counter.get("LLM_token", 0)
    tool_token_unit_price = _model_unit_price("text-davinci-003")  # 官方固定
    total_tokens = planner_tokens + solver_tokens + llm_tokens
    token_cost = planner_price * planner_tokens + solver_price * solver_tokens \
        + tool_token_unit_price * llm_tokens
    tool_cost = tool_counter.get("Google", 0) * 0.01
    return {
        "plans": plans,
        "planner_evidences": planner_evidences,
        "worker_calls": worker_calls,
        "worker_evidences": worker_evidences,
        "worker_log": worker_log,
        "output": solver_output,
        "tool_usage": tool_counter,
        "steps": len(plans) + 1,
        "total_tokens": total_tokens,
        "token_cost": token_cost,
        "tool_cost": tool_cost,
        "total_cost": token_cost + tool_cost,
    }


def _find_vars(text):
    """官方 re.findall(r"#E\\d+", ...)：按出现序、含重复。"""
    import re
    return re.findall(r"#E\d+", text)
