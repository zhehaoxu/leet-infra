"""ag-adapt-decomp-ledger 的独立 oracle 与检查点。

oracle 走另一条实现路径：执行器用 `while` 游标、表达式解析用递归下降、账本用扁平
列表累加，不使用参考解里的任何函数。
"""

from __future__ import annotations

import copy
import random
import re

_PLAIN = ("Nothing happens.", "OK.")


# --------------------------------------------------------------------------- oracle
def _o_exec(task, ex, max_runs, max_patience):
    if task not in ex:
        raise KeyError(task)
    script = list((ex[task] or {}).get("steps") or [])
    cursor = 0
    used = 0
    hist = []
    text = []
    ok = term = fin = False
    rw = 0
    bored = 0
    while cursor < len(script) and used < max_runs - 1:
        cur = script[cursor]
        cursor += 1
        used += 1
        act = cur["act"]
        obs = cur.get("obs", "Nothing happens.")
        step_rw = int(cur.get("reward", 0))
        fin = bool(cur.get("done", False))
        if act.startswith("think:"):
            obs = "OK."
            low = act.lower()
            if "task completed!" in low:
                fin, ok = True, True
            if "task failed!" in low:
                fin, ok = True, False
        else:
            hist.append(act)
        if obs in _PLAIN:
            bored += 1
            if bored >= max_patience:
                term = True
                break
        else:
            bored = 0
        text.append(" %s\n%s\n>" % (act, obs))
        if step_rw:
            rw, ok, term = step_rw, True, False
        if fin:
            break
    if not fin:
        ok, term, rw = False, True, 0
    return ok, rw, term, used, hist, "".join(text)


def _o_expr(text):
    toks = re.findall(r"Step \d+|AND|OR|\(|\)", text or "")
    pos = [0]

    def seq(depth):
        node = {}
        while pos[0] < len(toks):
            tok = toks[pos[0]]
            if tok == "(":
                pos[0] += 1
                node.setdefault("steps", []).append(seq(depth + 1))
            elif tok == ")":
                if not depth:
                    raise ValueError("unbalanced paren")
                pos[0] += 1
                return node
            elif tok in ("AND", "OR"):
                node["logic"] = tok
                pos[0] += 1
            else:
                node.setdefault("steps", []).append(int(tok.split()[1]))
                pos[0] += 1
        return node

    return seq(0)


def _o_fill(lookup, node):
    if "steps" not in node:
        raise ValueError("no steps")
    out = {}
    if "logic" in node:
        out["logic"] = node["logic"]
    out["steps"] = [
        lookup[s] if isinstance(s, int) else _o_fill(lookup, s) for s in node["steps"]
    ]
    return out


def _o_plan(plan):
    args = []
    logic = None
    for line in (plan or "").split("\n"):
        if line.startswith("Step"):
            args.append(re.sub(r"Step \d+: ", "", line))
        if "execution order" in line.lower():
            logic = line.split(": ")[-1]
    try:
        return _o_fill({i: a for i, a in enumerate(args, 1)}, _o_expr(logic))
    except Exception:  # noqa: BLE001
        return {"steps": args, "logic": "AND"}


def _o_salient(comp):
    body = comp.rstrip("\n>")
    if "Task completed!" not in body:
        return ""
    rows = body.split("\n")[:-2]
    while len(rows) > 1 and "think:" in rows[-2]:
        rows = rows[:-2]
    if len(rows) < 2:
        return ""
    return rows[-2].replace("Task completed!", "") + "\n" + rows[-1]


def _o_view(rec):
    direct = rec["direct"]
    return {
        "id": rec["id"],
        "task": rec["task"],
        "depth": rec["depth"],
        "replayed": rec["replayed"],
        "children": list(rec["children"]),
        "stop": rec["stop"],
        "success": rec["success"],
        "reward": rec["reward"],
        "plan": rec["plan"],
        "direct": None if direct is None else {
            "success": direct["success"],
            "reward": direct["reward"],
            "terminate": direct["terminate"],
            "turns": direct["turns"],
            "history": list(direct["history"]),
        },
    }


def _oracle(task, ex, pl, max_depth=3, max_runs=20, max_patience=5,
            info_prop_mode="last-step-last-act"):
    st = {"nodes": [], "runs": 0, "cp": []}

    def run(cur_task, depth, past):
        rec = {
            "id": len(st["nodes"]), "task": copy.deepcopy(cur_task), "depth": depth,
            "replayed": len(st["cp"]), "children": [], "direct": None, "plan": None,
            "stop": None, "success": False, "reward": 0,
        }
        st["nodes"].append(rec)

        if isinstance(cur_task, str):
            ok, rw, term, used, hist, text = _o_exec(cur_task, ex, max_runs, max_patience)
            st["runs"] += used
            rec["direct"] = {"success": ok, "reward": rw, "terminate": term,
                             "turns": used, "history": hist}
            running = ""
            if ok or depth >= max_depth:
                if ok:
                    st["cp"].extend(hist)
                running += text + "\n"
                info = past
                if ok:
                    info = (_o_salient(running) if info_prop_mode == "last-step-last-act"
                            else (past + "\n" + _o_salient(running)).lstrip("\n"))
                    rec["stop"] = "success"
                else:
                    rec["stop"] = "depth-cap"
                rec["success"], rec["reward"] = ok, rw
                return ok, rw, term, running, depth, info, rec["id"]
            if cur_task not in pl:
                raise KeyError(cur_task)
            parsed = _o_plan(pl[cur_task])
            st["runs"] += 1
            rec["plan"] = copy.deepcopy(parsed)
            if len(parsed["steps"]) == 1:
                only = parsed["steps"][0]
                if isinstance(only, str):
                    parsed = {"steps": [only], "logic": "AND"}
                elif isinstance(only, dict):
                    parsed = dict(only)
                    parsed.setdefault("logic", "AND")
            depth += 1
        else:
            parsed = copy.deepcopy(cur_task)

        if not isinstance(parsed.get("steps"), list):
            raise ValueError("bad plan")
        if not parsed["steps"]:
            raise ValueError("empty plan")
        parsed.setdefault("logic", "AND")
        rec["plan"] = copy.deepcopy(parsed)
        cap = str(parsed["logic"]).lower()

        running = ""
        info = past
        last = None
        for sub in parsed["steps"]:
            child = run(sub, depth, info)
            rec["children"].append(child[6])
            last = child
            if cap == "or" and child[0]:
                running += child[3] + "\n"
                if info_prop_mode == "last-step-last-act":
                    info = _o_salient(running)
                rec["stop"], rec["success"], rec["reward"] = "or-success", True, child[1]
                return True, child[1], child[2], running, depth, info, rec["id"]
            if child[0]:
                running += child[3] + "\n"
                if info_prop_mode == "last-step-last-act":
                    info = _o_salient(running)
            if cap == "and" and not child[0]:
                rec["stop"], rec["reward"] = "and-fail", child[1]
                return False, child[1], child[2], running, depth, info, rec["id"]

        rec["stop"] = "children-done"
        rec["success"], rec["reward"] = last[0], last[1]
        return last[0], last[1], last[2], running, depth, info, rec["id"]

    ok, rw, term, comp, depth, info, _nid = run(task, 1, "")
    return {
        "nodes": [_o_view(n) for n in st["nodes"]],
        "runs": st["runs"],
        "checkpoint": list(st["cp"]),
        "completion": comp,
        "info_prop": info,
        "success": ok,
        "reward": rw,
        "terminate": term,
        "depth": depth,
    }


# --------------------------------------------------------------------------- helpers
def _st(act, obs="OK.", reward=0, done=False):
    return {"act": act, "obs": obs, "reward": reward, "done": done}


def _view(r):
    return {
        "nodes": [_o_view(n) for n in r["nodes"]],
        "runs": r["runs"],
        "checkpoint": list(r["checkpoint"]),
        "completion": r["completion"],
        "info_prop": r["info_prop"],
        "success": r["success"],
        "reward": r["reward"],
        "terminate": r["terminate"],
        "depth": r["depth"],
    }


def _same(solve, task, ex, pl, **kw):
    got = _view(solve(copy.deepcopy(task), copy.deepcopy(ex), copy.deepcopy(pl), **kw))
    want = _oracle(copy.deepcopy(task), copy.deepcopy(ex), copy.deepcopy(pl), **kw)
    assert got == want, "ledger mismatch\ngot : %r\nwant: %r" % (got, want)
    return got


def _stops(r):
    return [(n["stop"], n["success"]) for n in r["nodes"]]


_ROOT = {"steps": [_st("look", "Nothing happens.")]}
_OK = {"steps": [_st("go to a 1", "You arrive."),
                 _st("think: Task completed!", done=True)]}


def _ok(act):
    """造一个「走一步再宣布完成」的成功脚本，动作名可区分。"""
    return {"steps": [_st(act, "You arrive."), _st("think: Task completed!", done=True)]}


# --------------------------------------------------------------------------- checks
def check_direct_success(solve):
    """根任务直接执行成功：不规划、不加深度，completion 逐段拼接。"""
    ex = {"t": {"steps": [
        _st("think: I should look around"),
        _st("look", "You see a mug."),
        _st("think: Task completed!", done=True),
    ]}}
    r = _same(solve, "t", ex, {})
    assert r["success"] is True
    assert r["reward"] == 0                     # think 完成不给 reward
    assert r["depth"] == 1
    assert r["runs"] == 3
    assert r["completion"] == (
        " think: I should look around\nOK.\n> look\nYou see a mug.\n> "
        "think: Task completed!\nOK.\n>\n")
    node = r["nodes"][0]
    assert node["direct"]["turns"] == 3
    assert node["direct"]["history"] == ["look"]      # think 不进 history
    assert node["stop"] == "success"
    assert r["checkpoint"] == ["look"]
    assert r["info_prop"] == "> look\nYou see a mug."

    # 真实动作 + reward 收尾：reward 记 1，completion 里没有 think
    ex = {"t": {"steps": [_st("go to a 1", "You arrive."),
                          _st("put mug 1 in/on b 1", "You put it.", reward=1, done=True)]}}
    r = _same(solve, "t", ex, {})
    assert (r["success"], r["reward"], r["runs"]) == (True, 1, 2)
    assert r["completion"] == " go to a 1\nYou arrive.\n> put mug 1 in/on b 1\nYou put it.\n>\n"
    assert r["info_prop"] == ""                # 没有字面 Task completed!


def check_as_needed_depth(solve):
    """先执行、失败才分解；到深度上限仍然执行一次，且不需要 plan。"""
    ex = {"t": _ROOT, "a": {"steps": [_st("look", "You see nothing.")]}}
    pl = {"t": "Step 1: a\nExecution order: Step 1"}

    # max_depth=1：一次分解都不发生（根直接执行后返回）
    r = _same(solve, "t", ex, {}, max_depth=1)
    assert [n["task"] for n in r["nodes"]] == ["t"]
    assert r["nodes"][0]["stop"] == "depth-cap"
    assert (r["success"], r["runs"]) == (False, 1)

    # max_depth=2：根分解，子任务在 depth 2 只执行、不规划（pl 里没有 a 也不报错）
    r = _same(solve, "t", ex, pl, max_depth=2)
    assert [(n["id"], n["depth"], n["stop"]) for n in r["nodes"]] == [
        (0, 1, "and-fail"), (1, 2, "depth-cap")]
    assert r["runs"] == 3                       # 根 1 步 + 1 次规划 + 子 1 步
    assert r["success"] is False


def check_patience(solve):
    """连续「没进展」到阈值即中断；触发那一步不写进 completion，reward 被丢弃。"""
    script = [_st("look", "Nothing happens.")] * 6 + [_st("think: Task completed!", done=True)]
    r = _same(solve, "t", {"t": {"steps": script}}, {}, max_depth=1)
    assert r["terminate"] is True
    assert r["success"] is False and r["reward"] == 0
    assert r["nodes"][0]["direct"]["turns"] == 5
    assert r["completion"].count("Nothing happens.") == 4    # 第 5 次没进文本
    assert r["completion"] == (" look\nNothing happens.\n> look\nNothing happens.\n> "
                               "look\nNothing happens.\n> look\nNothing happens.\n>\n")

    # 阈值改成 3 立刻生效（用 == 还是 >= 在这条路径上等价，但不能重置错）
    r = _same(solve, "t", {"t": {"steps": script}}, {}, max_depth=1, max_patience=3)
    assert r["nodes"][0]["direct"]["turns"] == 3
    assert r["completion"].count("Nothing happens.") == 2

    # 非「没进展」观测会把计数清零：OK. 与 Nothing happens. 交替不够连续就不断
    script = []
    for _ in range(3):
        script += [_st("look", "OK."), _st("look", "You see a mug.")]
    r = _same(solve, "t", {"t": {"steps": script}}, {}, max_depth=1, max_patience=2)
    assert r["nodes"][0]["direct"]["turns"] == 6
    assert r["terminate"] is True               # 脚本跑完仍未 done


def check_max_runs(solve):
    """`range(1, max_runs)` 少跑一步；跑不完就把 success/reward 一起抹掉。"""
    five = {"steps": [_st("look", "You see a cup."), _st("go to a 1", "You arrive."),
                      _st("take cup 1", "You take it."), _st("go to b 1", "You arrive."),
                      _st("put cup 1 in/on b 1", "You put it.", reward=1, done=True)]}
    r = _same(solve, "t", {"t": five}, {}, max_depth=1, max_runs=5)
    assert r["nodes"][0]["direct"]["turns"] == 4
    assert r["success"] is False and r["reward"] == 0
    assert r["nodes"][0]["direct"]["history"][-1] == "go to b 1"

    r = _same(solve, "t", {"t": five}, {}, max_depth=1, max_runs=6)
    assert r["nodes"][0]["direct"]["turns"] == 5
    assert (r["success"], r["reward"]) == (True, 1)

    # max_runs=1 → 一步都不跑
    r = _same(solve, "t", {"t": five}, {}, max_depth=1, max_runs=1)
    assert r["nodes"][0]["direct"]["turns"] == 0
    assert r["completion"] == "\n"          # 一段都没跑，只留下拼接用的换行
    assert (r["success"], r["terminate"]) == (False, True)


def check_reward_semantics(solve):
    """reward 置成功并清 terminate，但脚本自然跑完时连 reward 一起丢。"""
    # 有 reward 但没 done → 继续跑，最后一步没 done → 全丢
    ex = {"t": {"steps": [_st("take mug 1", "You take it.", reward=1),
                          _st("look", "OK.")]}}
    r = _same(solve, "t", ex, {}, max_depth=1)
    assert (r["success"], r["reward"], r["terminate"]) == (False, 0, True)

    # reward + done → 成功
    ex = {"t": {"steps": [_st("take mug 1", "You take it.", reward=1, done=True)]}}
    r = _same(solve, "t", ex, {}, max_depth=1)
    assert (r["success"], r["reward"], r["terminate"]) == (True, 1, False)

    # 先 reward 后连续「没进展」→ 中断把 success 压回 False
    ex = {"t": {"steps": [_st("take mug 1", "You take it.", reward=1)]
                + [_st("look", "OK.")] * 3}}
    r = _same(solve, "t", ex, {}, max_depth=1, max_patience=2)
    assert (r["success"], r["reward"], r["terminate"]) == (False, 0, True)


def check_think_and_failed(solve):
    """think 里的 completed/failed 判定不分大小写，但 info 提取只要字面写法。"""
    ex = {"t": {"steps": [_st("go to a 1", "You arrive."),
                          _st("think: task completed!", done=True)]}}
    r = _same(solve, "t", ex, {}, max_depth=1)
    assert (r["success"], r["info_prop"]) == (True, "")     # 小写 → success 但 info 空

    ex = {"t": {"steps": [_st("go to a 1", "You arrive."),
                          _st("think: Task failed!", done=True)]}}
    r = _same(solve, "t", ex, {}, max_depth=1)
    assert (r["success"], r["reward"], r["info_prop"]) == (False, 0, "")
    assert r["checkpoint"] == []                            # 失败不收 action


def check_plan_fallback(solve):
    """解析失败一律退化成「平铺 + AND」，括号里的嵌套整个丢掉。"""
    ex = {"t": _ROOT, "a": _OK, "b": _OK}

    # 没有 execution order 行
    r = _same(solve, "t", ex, {"t": "Step 1: a\nStep 2: b"}, max_depth=2)
    assert r["nodes"][0]["plan"] == {"steps": ["a", "b"], "logic": "AND"}
    assert [n["task"] for n in r["nodes"]] == ["t", "a", "b"]

    # Step 编号越界
    r = _same(solve, "t", ex, {"t": "Step 1: a\nStep 2: b\nExecution order: (Step 1 AND Step 3)"},
              max_depth=2)
    assert r["nodes"][0]["plan"] == {"steps": ["a", "b"], "logic": "AND"}

    # 右括号多一个：官方在 stack.pop() 上抛错 → 兜底 AND（即使表达式里写的是 OR）
    r = _same(solve, "t", ex,
              {"t": "Step 1: a\nStep 2: b\nExecution order: Step 1 OR Step 2)"},
              max_depth=2)
    assert r["nodes"][0]["plan"] == {"steps": ["a", "b"], "logic": "AND"}
    assert [n["task"] for n in r["nodes"]] == ["t", "a", "b"]

    # 只有执行顺序、没有 Step 行 → 空计划直接报错
    try:
        solve("t", {"t": _ROOT}, {"t": "Execution order: Step 1"})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty plan")


def check_logic_parsing(solve):
    """同一层多个逻辑词后写覆盖先写；括号子式变成嵌套 dict。"""
    ex = {"t": _ROOT, "a": _OK, "b": _OK, "c": _OK}

    r = _same(solve, "t", ex,
              {"t": "Step 1: a\nStep 2: b\nStep 3: c\n"
                    "Execution order: Step 1 AND Step 2 OR Step 3"}, max_depth=2)
    assert r["nodes"][0]["plan"]["logic"] == "OR"            # 后写的覆盖先写的
    assert [n["task"] for n in r["nodes"]] == ["t", "a"]       # 首个成功即返回

    r = _same(solve, "t", ex,
              {"t": "Step 1: a\nStep 2: b\nStep 3: c\n"
                    "Execution order: (Step 1 AND (Step 2 OR Step 3))"}, max_depth=2)
    assert r["nodes"][0]["plan"] == {
        "steps": ["a", {"steps": ["b", "c"], "logic": "OR"}], "logic": "AND"}

    # 执行顺序行只要出现在**行内**即可（前缀文字不影响），且取 ": " 的最后一段
    r = _same(solve, "t", ex,
              {"t": "Step 1: a\nStep 2: b\nNote: execution order: (Step 1 OR Step 2)"},
              max_depth=2)
    assert r["nodes"][0]["plan"]["logic"] == "OR"
    assert [n["task"] for n in r["nodes"]] == ["t", "a"]

    # 执行顺序行大小写不敏感，但逻辑词只认大写：`or` 不是 token，整层没有逻辑词 → AND
    r = _same(solve, "t", ex,
              {"t": "Step 1: a\nStep 2: b\nEXECUTION ORDER: (Step 1 or Step 2)"},
              max_depth=2)
    assert r["nodes"][0]["plan"] == {"steps": ["a", "b"], "logic": "AND"}


def check_and_or_short_circuit(solve):
    """AND 遇失败立刻返回、OR 遇成功立刻返回，剩下的兄弟节点不访问。"""
    fail = {"steps": [_st("go to a 1", "You arrive."),
                      _st("think: Task failed!", done=True)]}
    ex = {"t": _ROOT, "a": fail, "b": _OK, "c": _OK}

    r = _same(solve, "t", ex,
              {"t": "Step 1: a\nStep 2: b\nStep 3: c\n"
                    "Execution order: (Step 1 AND Step 2 AND Step 3)"}, max_depth=2)
    assert [n["task"] for n in r["nodes"]] == ["t", "a"]
    assert _stops(r) == [("and-fail", False), ("depth-cap", False)]
    assert r["checkpoint"] == [] and r["success"] is False

    r = _same(solve, "t", ex,
              {"t": "Step 1: a\nStep 2: b\nStep 3: c\n"
                    "Execution order: (Step 1 OR Step 2 OR Step 3)"}, max_depth=2)
    assert [n["task"] for n in r["nodes"]] == ["t", "a", "b"]
    assert _stops(r) == [("or-success", True), ("depth-cap", False), ("success", True)]
    assert r["checkpoint"] == ["go to a 1"]
    # 命中后不再跑 c，但 a 的失败动作已经留在 checkpoint 里
    assert "go to c 1" not in r["completion"]


def check_group_subplan(solve):
    """括号子式是已解析计划：不试执行器、不规划、不加深度。"""
    ex = {"t": _ROOT, "a": _ok("go to a 1"), "b": _ok("go to b 1"), "c": _ok("go to c 1")}
    r = _same(solve, "t", ex,
              {"t": "Step 1: a\nStep 2: b\nStep 3: c\n"
                    "Execution order: ((Step 1 AND Step 2) AND Step 3)"}, max_depth=2)
    group = r["nodes"][1]
    assert group["task"] == {"steps": ["a", "b"], "logic": "AND"}
    assert group["direct"] is None                  # 没有执行器尝试
    assert group["depth"] == 2                      # 与兄弟同深度
    assert group["stop"] == "children-done"
    assert [n["depth"] for n in r["nodes"]] == [1, 2, 2, 2, 2]
    # 根 1 步 + 1 次规划 + a/b/c 各 2 步（组本身不花轮次）
    assert r["runs"] == 8
    assert r["checkpoint"] == ["go to a 1", "go to b 1", "go to c 1"]


def check_checkpoint_and_replay(solve):
    """checkpoint 只收成功节点的动作；后续节点按 checkpoint 长度复放。"""
    ex = {
        "t": {"steps": [_st("look", "Nothing happens.")]},
        "a": {"steps": [_st("look", "You see nothing.")]},
        "a1": _OK,
        "a2": {"steps": [_st("go to fridge 1", "You arrive."),
                         _st("think: Task completed!", done=True)]},
    }
    pl = {"t": "Step 1: a\nExecution order: Step 1",
          "a": "Step 1: a1\nStep 2: a2\nExecution order: (Step 1 AND Step 2)"}
    r = _same(solve, "t", ex, pl, max_depth=3)
    assert [(n["depth"], n["replayed"]) for n in r["nodes"]] == [
        (1, 0), (2, 0), (3, 0), (3, 1)]
    assert r["checkpoint"] == ["go to a 1", "go to fridge 1"]
    assert r["success"] is True
    assert r["depth"] == 2                          # 根返回的是自己递增后的深度
    assert r["runs"] == 8


def check_info_propagation(solve):
    """last-step 模式在每处成功都刷新；all-step 只在叶子成功时刷新一次。"""
    ex = {
        "t": {"steps": [_st("look", "Nothing happens.")]},
        "a": {"steps": [_st("look", "You see nothing.")]},
        "a1": _OK,
        "a2": _OK,
    }
    pl = {"t": "Step 1: a\nExecution order: Step 1",
          "a": "Step 1: a1\nStep 2: a2\nExecution order: (Step 1 AND Step 2)"}

    r = _same(solve, "t", ex, pl, max_depth=3)
    assert r["info_prop"] == " go to a 1\nYou arrive."       # 最后一个成功子节点

    r = _same(solve, "t", ex, pl, max_depth=3, info_prop_mode="all-step-last-act")
    assert r["info_prop"] == ""                              # 子节点成功不刷新

    # 抽出的 act 行里真带 `Task completed!` 字样时会被剥掉（只剥这一串）
    ex = {"t": {"steps": [_st("say Task completed!", "fine"),
                          _st("look", "You see a cup.", reward=1, done=True)]}}
    r = _same(solve, "t", ex, {}, max_depth=1)
    assert r["info_prop"] == " say \nfine"


def check_default_args(solve):
    """默认 max_depth=3 / max_runs=20 / max_patience=5 / last-step。"""
    # 三层链：根 → a 失败 → a 分解为 a1/a2
    ex = {"t": _ROOT, "a": {"steps": [_st("look", "You see nothing.")]},
          "a1": _OK, "a2": _OK}
    pl = {"t": "Step 1: a\nExecution order: Step 1",
          "a": "Step 1: a1\nStep 2: a2\nExecution order: (Step 1 AND Step 2)"}
    r = _same(solve, "t", ex, pl)
    assert [n["depth"] for n in r["nodes"]] == [1, 2, 3, 3]
    assert r["success"] is True

    # 默认 max_runs=20 → 19 步跑得完
    steps = [_st("look", "You see a cup.")] * 18 + [_st("think: Task completed!", done=True)]
    r = _same(solve, "t", {"t": {"steps": steps}}, {}, max_depth=1)
    assert r["nodes"][0]["direct"]["turns"] == 19
    assert r["success"] is True

    # 默认 max_patience=5
    steps = [_st("look", "OK.")] * 4 + [_st("think: Task completed!", done=True)]
    r = _same(solve, "t", {"t": {"steps": steps}}, {}, max_depth=1)
    assert r["success"] is True                       # 4 次还没到 5，能走到 think

    # 默认 info 模式
    r = _same(solve, "t", {"t": _OK}, {})
    assert r["info_prop"] == " go to a 1\nYou arrive."


def check_errors(solve):
    """域约束：缺脚本、缺计划、坏计划、坏参数。"""
    def _bad(fn):
        try:
            fn()
        except ValueError:
            return True
        except Exception as exc:                     # noqa: BLE001
            raise AssertionError("wrong error: %r" % (exc,))
        raise AssertionError("expected ValueError")

    assert _bad(lambda: solve("zzz", {}, {}))
    assert _bad(lambda: solve("t", {"t": _ROOT}, {}, max_depth=3))
    assert _bad(lambda: solve("t", {"t": _ROOT}, {"t": "Step 1: a"}))
    assert _bad(lambda: solve("t", {"t": _ROOT}, {}, max_depth=0))
    assert _bad(lambda: solve("t", {"t": _ROOT}, {}, max_patience=0))
    assert _bad(lambda: solve("", {"": _ROOT}, {}))
    assert _bad(lambda: solve("t", {"t": {"steps": [{"obs": "OK."}]}}, {}))
    assert _bad(lambda: solve("t", {"t": _OK}, {}, info_prop_mode="nope"))


def check_random_cross(solve):
    """随机剧本交叉验证：与独立 oracle 的整本账逐字段比对。"""
    rng = random.Random(20261102)
    acts = ["look", "go to a 1", "go to b 1", "take cup 1", "put cup 1 in/on b 1",
            "think: hmm", "think: Task completed!", "think: Task failed!",
            "think: task completed!"]
    obs = ["You arrive.", "Nothing happens.", "OK.", "You see a cup.", "You put it."]

    def make_ex(names, k):
        ex = {}
        for name in names[:k]:
            n = rng.randint(0, 6)
            steps = []
            for _ in range(n):
                steps.append(_st(rng.choice(acts), rng.choice(obs),
                                 rng.choice([0, 0, 0, 1]), rng.random() < 0.4))
            ex[name] = {"steps": steps}
        return ex

    def expr(lo, hi):
        if lo == hi:
            return "Step %d" % lo
        cut = rng.randint(lo, hi - 1)
        out = "%s %s %s" % (expr(lo, cut), rng.choice(["AND", "OR"]), expr(cut + 1, hi))
        return "(" + out + ")" if rng.random() < 0.4 else out

    def make_pl(names, k):
        pl = {}
        for i, name in enumerate(names[:k]):
            slots = names[i + 1:k] or names[:1]
            cnt = rng.randint(1, min(3, len(slots)))
            picked = [rng.choice(slots) for _ in range(cnt)]
            text = "\n".join("Step %d: %s" % (j + 1, s) for j, s in enumerate(picked))
            if rng.random() < 0.2:
                pl[name] = text                       # 无 execution order → 兜底
            elif rng.random() < 0.15:
                pl[name] = text + "\nExecution order: Step 99"   # 越界 → 兜底
            else:
                pl[name] = text + "\nExecution order: " + expr(1, cnt)
        return pl

    names = ["t%d" % i for i in range(9)]
    for _ in range(40):
        k = rng.randint(1, 6)
        ex = make_ex(names, k)
        pl = make_pl(names, k)
        kw = {
            "max_depth": rng.randint(1, 3),
            "max_runs": rng.randint(1, 7),
            "max_patience": rng.randint(1, 4),
            "info_prop_mode": rng.choice(["last-step-last-act", "all-step-last-act"]),
        }
        if ex[names[0]].get("steps") and rng.random() < 0.5:
            ex[names[0]]["steps"] = ex[names[0]]["steps"][:rng.randint(0, 3)]
        _same(solve, names[0], ex, pl, **kw)


def check_example(solve):
    """样例：根失败 → 规划成 AND 两步 → 都成功。"""
    ex = {
        "t": _ROOT,
        "wash the mug": {"steps": [_st("go to sink 1", "You arrive."),
                                   _st("think: Task completed!", done=True)]},
        "put the mug": {"steps": [_st("go to cabinet 1", "You arrive."),
                                  _st("put mug 1 in/on cabinet 1", "You put it.", reward=1,
                                      done=True)]},
    }
    pl = {"t": "Step 1: wash the mug\nStep 2: put the mug\n"
               "Execution order: (Step 1 AND Step 2)"}
    r = solve("t", ex, pl)
    assert r["success"] is True
    assert r["reward"] == 1
    assert [n["task"] for n in r["nodes"]] == ["t", "wash the mug", "put the mug"]
    assert r["checkpoint"] == ["go to sink 1", "go to cabinet 1", "put mug 1 in/on cabinet 1"]
    assert r["runs"] == 6


check_example.example = True
