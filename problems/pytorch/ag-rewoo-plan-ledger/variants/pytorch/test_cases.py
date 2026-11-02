import math
import random
import re

# ============ 独立 oracle（与 solution 不同构的实现） ============

_PRICE = {"text-davinci-003": 0.00002, "gpt-3.5-turbo": 0.000002,
          "gpt-4": 0.00003, "tloen/alpaca-lora-7b": 0.0,
          "rewoo/planner_7B": 0.0}


def _price(model):
    if model not in _PRICE:
        raise ValueError("Model not found")
    return _PRICE[model]


def _oracle(planner_response, worker_outputs, solver_output,
            planner_tokens=0, solver_tokens=0, tools=("Google", "LLM"),
            planner_model="text-davinci-003", solver_model="text-davinci-003"):
    if not isinstance(planner_response, str):
        raise ValueError("planner_response must be a str")
    if not isinstance(worker_outputs, dict):
        raise ValueError("worker_outputs must be a dict")
    if not isinstance(solver_output, str):
        raise ValueError("solver_output must be a str")
    if not isinstance(tools, (list, tuple)) or \
            not all(isinstance(t, str) for t in tools):
        raise ValueError("tools must be a list of tool names")
    p_price, s_price = _price(planner_model), _price(solver_model)

    lines = planner_response.splitlines()
    plans = [ln for ln in lines if ln[:5] == "Plan:"]

    evid = {}
    for ln in lines:
        if ln.startswith("#") and len(ln) >= 3 and ln[1] == "E" \
                and ln[2].isdigit():
            if ln.count("=") == 0:
                raise ValueError(f"malformed evidence line: {ln!r}")
            name, _, call = ln.partition("=")
            name, call = name.strip(), call.strip()
            evid[name] = call if len(name) == 3 else "No evidence found"

    ev_worker, calls, usage = {}, {}, {}
    for name in evid:  # dict 插入序 = 首次出现序
        call = evid[name]
        if "[" not in call:
            ev_worker[name] = call
            calls[name] = None
            continue
        head, rest = call.split("[", 1)
        arg = rest[:-1]
        for var in re.findall(r"#E\d+", arg):
            if var in ev_worker:
                arg = arg.replace(var, "[" + ev_worker[var] + "]")
        calls[name] = arg
        if head in tools:
            if name not in worker_outputs:
                raise ValueError(f"missing worker output for {name}")
            ev_worker[name] = worker_outputs[name]
            if head == "Google":
                usage["Google"] = usage.get("Google", 0) + 1
            elif head == "LLM":
                usage["LLM_token"] = usage.get("LLM_token", 0) \
                    + len(arg + ev_worker[name]) // 4
        else:
            ev_worker[name] = "No evidence found"

    log = ""
    for i, plan in enumerate(plans):
        key = f"#E{i + 1}"
        if key not in ev_worker:
            raise ValueError(f"plan {i} has no matching evidence {key}")
        log += f"{plan}\nEvidence:\n{ev_worker[key]}\n"

    llm_tok = usage.get("LLM_token", 0)
    total_tokens = planner_tokens + solver_tokens + llm_tok
    token_cost = p_price * planner_tokens + s_price * solver_tokens \
        + _price("text-davinci-003") * llm_tok
    tool_cost = usage.get("Google", 0) * 0.01
    return {"plans": plans, "planner_evidences": evid,
            "worker_calls": calls, "worker_evidences": ev_worker,
            "worker_log": log, "output": solver_output,
            "tool_usage": usage, "steps": len(plans) + 1,
            "total_tokens": total_tokens, "token_cost": token_cost,
            "tool_cost": tool_cost, "total_cost": token_cost + tool_cost}


# ============ 比较工具 ============

def _eq(a, b, path="root"):
    if isinstance(a, float) or isinstance(b, float):
        assert isinstance(a, (int, float)) and isinstance(b, (int, float)), \
            f"{path}: {a!r} vs {b!r}"
        assert math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-9), \
            f"{path}: {a!r} vs {b!r}"
        return
    if type(a) is not type(b) and not (isinstance(a, (int, float))
                                      and isinstance(b, (int, float))):
        raise AssertionError(f"{path}: type {type(a)} vs {type(b)}")
    if isinstance(a, dict):
        assert set(a) == set(b), f"{path}: keys {set(a)} vs {set(b)}"
        for k in a:
            _eq(a[k], b[k], f"{path}.{k}")
        return
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: len {len(a)} vs {len(b)}"
        for i, (va, vb) in enumerate(zip(a, b)):
            _eq(va, vb, f"{path}[{i}]")
        return
    assert a == b, f"{path}: {a!r} vs {b!r}"


def _deep_copy(v):
    if isinstance(v, dict):
        return {k: _deep_copy(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_deep_copy(x) for x in v]
    return v


def _compare(solve, planner_response, worker_outputs, solver_output, **kw):
    got = solve(planner_response, _deep_copy(worker_outputs), solver_output,
                **kw)
    exp = _oracle(planner_response, _deep_copy(worker_outputs), solver_output,
                  **kw)
    _eq(got, exp)
    return got


# ============ 手算锚点 ============

def check_basic_chain(solve):
    """Plan 解析 + Google/LLM 账 + worker_log 下标配对 + 成本账。"""
    resp = ("Plan: Search for the capital\n"
            "#E1 = Google[capital of France]\n"
            "Plan: Summarize with LLM\n"
            "#E2 = LLM[Summarize #E1 in one line]\n")
    out = _compare(solve, resp, {"#E1": "Paris is the capital of France.",
                                 "#E2": "It is Paris."}, "It is Paris.",
                   planner_tokens=120, solver_tokens=30)
    assert out["plans"] == ["Plan: Search for the capital",
                            "Plan: Summarize with LLM"]
    assert out["planner_evidences"] == {
        "#E1": "Google[capital of France]",
        "#E2": "LLM[Summarize #E1 in one line]"}
    # 替换：原括号被 split 吃掉，插入文本重新包 []
    assert out["worker_calls"]["#E1"] == "capital of France"
    exp_call = "Summarize [Paris is the capital of France.] in one line"
    assert out["worker_calls"]["#E2"] == exp_call
    assert out["worker_evidences"]["#E1"] == "Paris is the capital of France."
    assert out["tool_usage"] == {"Google": 1,
                                 "LLM_token": len(exp_call + "It is Paris.") // 4}
    assert out["worker_log"] == (
        "Plan: Search for the capital\nEvidence:\n"
        "Paris is the capital of France.\n"
        "Plan: Summarize with LLM\nEvidence:\nIt is Paris.\n")
    assert out["steps"] == 3
    assert out["total_tokens"] == 150 + out["tool_usage"]["LLM_token"]
    assert math.isclose(out["token_cost"], 0.00002 * 150
                        + 0.00002 * out["tool_usage"]["LLM_token"],
                        abs_tol=1e-9)
    assert out["tool_cost"] == 0.01
    assert math.isclose(out["total_cost"],
                        out["token_cost"] + 0.01, abs_tol=1e-9)


def check_multidigit_quirk(solve):
    """len('#Ed')==3 才有效：#E12 的 Google 调用被丢弃记 No evidence found，
    且该字符串作为 pass-through 证据参与后续替换（须先于引用者出现）。"""
    resp = ("#E12 = Google[query]\n"
            "Plan: b\n#E1 = LLM[Use #E12]\n")
    out = _compare(solve, resp, {"#E1": "done"}, "done")
    assert out["planner_evidences"]["#E12"] == "No evidence found"
    assert out["worker_evidences"]["#E12"] == "No evidence found"
    assert out["worker_calls"]["#E12"] is None      # 无 '[' → pass-through
    assert out["worker_calls"]["#E1"] == "Use [No evidence found]"
    assert out["tool_usage"] == {"LLM_token":
                                 len("Use [No evidence found]done") // 4}
    assert "Google" not in out["tool_usage"]        # Google 调用被丢弃
    assert out["plans"] == ["Plan: b"]


def check_forward_reference(solve):
    """前向引用：证据未算出 → 原样保留占位符。"""
    resp = "Plan: p\n#E1 = LLM[Compare #E2 and #E3]\n#E2 = Google[q]\n#E3 = Google[r]\n"
    out = _compare(solve, resp, {"#E1": "x", "#E2": "cats", "#E3": "dogs"},
                   "x", tools=("Google", "LLM"))
    assert out["worker_calls"]["#E1"] == "Compare #E2 and #E3"  # 原样保留
    assert out["worker_evidences"]["#E2"] == "cats"
    assert out["worker_evidences"]["#E3"] == "dogs"
    # LLM_token 用未替换的字面调用串
    assert out["tool_usage"]["LLM_token"] == len("Compare #E2 and #E3" + "x") // 4


def check_double_substitution(solve):
    """replace-all 会命中先前替换插入文本中的占位符（官方 quirk）。"""
    resp = ("Plan: p\n#E1 = Quote #E2 line\n"
            "#E2 = Google[q]\n#E3 = LLM[#E1 and #E2]\n")
    out = _compare(solve, resp, {"#E2": "cats", "#E3": "ans"}, "ans")
    # 先替换 #E1 → "[Quote #E2 line]"，再替换 #E2 时连插入文本里的 #E2 一起换
    assert out["worker_calls"]["#E3"] == "[Quote [cats] line] and [cats]"
    assert out["worker_evidences"]["#E1"] == "Quote #E2 line"


def check_split_quirks(solve):
    """split('[',1) 后盲删末字符；工具名带尾空格（'Google ['）即未知工具
    （'=' 右侧先被整体 strip，前导空格 quirk 不存在）。"""
    resp = ("Plan: p\n#E1 = Google[foo][bar\n"
            "Plan: q\n#E2 = Google [x]\n")
    out = _compare(solve, resp, {"#E1": "ok"}, "z")
    assert out["worker_calls"]["#E1"] == "foo][ba"          # 盲删 'r'
    assert out["worker_evidences"]["#E1"] == "ok"
    assert out["worker_calls"]["#E2"] == "x"                 # 替换仍执行
    assert out["worker_evidences"]["#E2"] == "No evidence found"  # 'Google ' 未知
    assert out["tool_usage"] == {"Google": 1}                # 只有 #E1 计入


def check_duplicate_and_noise(solve):
    """重复证据行后者覆盖；Plan:/#E 之外的行全部忽略。"""
    resp = ("noise line\n#E1 = Google[old]\n#E1 = Google[new]\n"
            "Plan: only\nanother noise\n")
    out = _compare(solve, resp, {"#E1": "res"}, "final")
    assert out["planner_evidences"] == {"#E1": "Google[new]"}
    assert out["worker_evidences"] == {"#E1": "res"}
    assert out["plans"] == ["Plan: only"]
    assert out["steps"] == 2
    assert out["worker_log"] == "Plan: only\nEvidence:\nres\n"


def check_cost_table(solve):
    """价格表逐项 + token 整除 //4 + Google 计数成本。"""
    resp = "Plan: p\n#E1 = LLM[hello]\nPlan: q\n#E2 = Google[g]\n#E3 = Google[h]\n"
    out = _compare(solve, resp, {"#E1": "1234567", "#E2": "a", "#E3": "b"},
                   "x", planner_tokens=1000, solver_tokens=500,
                   planner_model="gpt-4", solver_model="gpt-3.5-turbo",
                   tools=("Google", "LLM"))
    # len("hello"+"1234567")=12 → //4 = 3
    assert out["tool_usage"] == {"Google": 2, "LLM_token": 3}
    assert out["total_tokens"] == 1503
    assert math.isclose(out["token_cost"], 0.00003 * 1000 + 0.000002 * 500
                        + 0.00002 * 3, abs_tol=1e-12)
    assert out["tool_cost"] == 0.02
    # llama 价格 0
    out2 = _compare(solve, resp, {"#E1": "1234567", "#E2": "a", "#E3": "b"},
                    "x", planner_tokens=100, solver_tokens=100,
                    planner_model="tloen/alpaca-lora-7b",
                    solver_model="rewoo/planner_7B")
    assert out2["token_cost"] == 0.00002 * 3
    for bad_model in ["gpt-5", "davinci", ""]:
        try:
            solve("Plan: p\n#E1 = Google[q]\n", {"#E1": "ok"}, "x",
                  planner_model=bad_model)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for model price")


def check_worker_log_mismatch(solve):
    """Plan 行数与 #E{i+1} 证据缺失 → ValueError（官方 KeyError）。"""
    resp = "Plan: a\n#E1 = Google[q]\nPlan: b\n"
    try:
        solve(resp, {"#E1": "ok"}, "x")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for missing #E2")
    # 证据多于 Plan：多出的证据照算但不进 worker_log
    resp2 = "Plan: a\n#E1 = Google[q]\n#E2 = Google[r]\n"
    out = _compare(solve, resp2, {"#E1": "1", "#E2": "2"}, "x")
    assert len(out["plans"]) == 1
    assert out["tool_usage"]["Google"] == 2
    assert out["worker_log"].count("Plan:") == 1


def check_invalid_inputs(solve):
    good = "Plan: p\n#E1 = Google[q]\n"
    for bad in [
        lambda: solve(1, {}, "x"),                       # response 非字符串
        lambda: solve(good, [], "x"),                    # worker_outputs 非字典
        lambda: solve(good, {}, 3),                      # solver_output 非字符串
        lambda: solve(good, {}, "x", tools="Google"),    # tools 非列表
        lambda: solve(good, {}, "x", tools=[1]),         # 工具名非字符串
        lambda: solve(good, {}, "x", planner_tokens=-1),
        lambda: solve(good, {}, "x", planner_tokens=True),
        lambda: solve(good, {}, "x", solver_tokens=2.5),
        lambda: solve("Plan: p\n#E2\n", {}, "x"),        # 证据行无 '='
        lambda: solve("Plan: p\n#E\n", {}, "x"),         # 形残证据行
        lambda: solve(good, {}, "x"),                    # 缺 worker 输出
    ]:
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


# ============ 随机对拍 ============

_TOOLS_POOL = ["Google", "LLM", "Wikipedia"]
_UNKNOWN_TOOLS = ["Bing", "Calc"]
_ARG_WORDS = ["q", "sum", "data", "x y", ""]


def _gen_case(rng):
    tools = tuple(rng.sample(["Google", "LLM", "Wikipedia"],
                             rng.randint(1, 3)))
    n = rng.randint(1, 6)
    lines, outputs = [], {}
    index_pool = list(range(1, 13))
    used = []
    for _ in range(n):
        r = rng.random()
        if r < 0.25:
            lines.append(f"Plan: step {len(used) + 1} do work")
        elif r < 0.3:
            lines.append(rng.choice(["noise", "# note", ""]))
        else:
            idx = rng.choice(index_pool)
            e = f"#E{idx}"
            if idx >= 10:
                # 多位下标：解析层 len(e)==3 quirk，工具调用被丢弃
                lines.append(f"{e} = Google[q{idx}]")
                continue
            kind = rng.random()
            if kind < 0.25:
                lines.append(f"{e} = Extract point {idx}")
            elif kind < 0.9:
                tool = rng.choice(list(tools) + _UNKNOWN_TOOLS)
                nvars = rng.randint(0, 2)
                arg = rng.choice(_ARG_WORDS)
                for _v in range(nvars):
                    ref = rng.choice(index_pool)
                    arg += f" #E{ref}"
                lines.append(f"{e} = {tool}[{arg}]")
                if tool in tools:
                    outputs[e] = rng.choice(["res", "a b c d e", "z9"])
            else:
                lines.append(f"{e} = Google[q{idx}]")
                if "Google" in tools:
                    outputs[e] = rng.choice(["res", "ok"])
    # 补齐 Plan 行需要的证据（缺失时 solve/oracle 一致报错，随机里允许）
    response = "\n".join(lines)
    return response, outputs


def check_random(solve):
    rng = random.Random(7210721)
    for _ in range(200):
        response, outputs = _gen_case(rng)
        kw = dict(planner_tokens=rng.randint(0, 200),
                  solver_tokens=rng.randint(0, 200),
                  planner_model=rng.choice(
                      ["text-davinci-003", "gpt-4", "gpt-3.5-turbo",
                       "tloen/alpaca-lora-7b"]),
                  solver_model=rng.choice(
                      ["text-davinci-003", "gpt-4", "gpt-3.5-turbo"]),
                  tools=rng.choice([("Google", "LLM"),
                                    ("Google", "LLM", "Wikipedia"),
                                    ("LLM",)]))
        try:
            got = solve(response, _deep_copy(outputs), "ANS", **kw)
        except ValueError:
            # oracle 必须在同一输入上同样报错
            try:
                _oracle(response, _deep_copy(outputs), "ANS", **kw)
                raise AssertionError("solve raised but oracle did not")
            except ValueError:
                continue
        exp = _oracle(response, _deep_copy(outputs), "ANS", **kw)
        _eq(got, exp)


TEST_CASES = [
    ("basic_chain", check_basic_chain),
    ("multidigit_quirk", check_multidigit_quirk),
    ("forward_reference", check_forward_reference),
    ("double_substitution", check_double_substitution),
    ("split_quirks", check_split_quirks),
    ("duplicate_and_noise", check_duplicate_and_noise),
    ("cost_table", check_cost_table),
    ("worker_log_mismatch", check_worker_log_mismatch),
    ("invalid_inputs", check_invalid_inputs),
    ("random", check_random),
]
