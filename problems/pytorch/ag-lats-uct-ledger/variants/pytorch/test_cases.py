import math
import random

from copy import deepcopy


def _dc(x):
    return deepcopy(x)


# ---------------------------------------------------------------- oracle
# 独立实现（不 import solution），同官方 mcts.py 语义

def _oracle(root_solution, iterations, n, n_tests, plans,
            root_feedback="", root_reflection=""):
    nodes = [{"id": 0, "solution": root_solution, "parent": None,
              "children": [], "depth": 0, "visits": 0, "value": 0,
              "reflection": root_reflection,
              "test_feedback": root_feedback, "pref": None}]

    def uct(nd, w=1.0):
        if nd["visits"] == 0:
            return nd["value"]
        return (nd["value"] / nd["visits"]) + \
            w * math.sqrt(math.log(nd["pref"]["visits"]) / nd["visits"])

    iters = []
    is_solved = False
    solved_child = None

    for cur_iter in range(iterations):
        if is_solved:
            break
        if cur_iter >= len(plans):
            break
        plan = plans[cur_iter]
        node = nodes[0]
        path = [0]
        while node["children"]:
            if not node["children"]:
                break
            best = None
            for c in node["children"]:          # max 平分取先
                if best is None or uct(c) > uct(best):
                    best = c
            node = best
            path.append(node["id"])
        sims = list(plan["sims"])
        new_ids = []
        for sol in plan["solutions"][:n]:
            c = {"id": len(nodes), "solution": sol, "parent": node["id"],
                 "children": [], "depth": node["depth"] + 1,
                 "visits": 0, "value": 0, "reflection": "",
                 "test_feedback": "", "pref": node}
            node["children"].append(c)
            nodes.append(c)
            new_ids.append(c["id"])
        simulated = []
        reward_internal = 0.0
        reward_real = 0
        child = node["children"][-1] if node["children"] else None
        if child is None:
            raise ValueError("no child")
        for c in node["children"]:
            passing, feedback, refl, ev = sims.pop(0)
            simulated.append(c["id"])
            c["reflection"] = "" if passing else refl
            c["test_feedback"] = feedback
            if "Tested passed:" in feedback:
                ps = feedback.split("Tests failed:")[0]
                cnt = len([ln for ln in
                           ps.split("Tested passed:")[1].splitlines()
                           if ln.strip() != ""])
                reward_internal = cnt / n_tests
            else:
                reward_internal = 0
            if passing or cur_iter == iterations - 1:
                if ev:
                    reward_real = 1
                    is_solved = True
                    solved_child = c
                break
        acc_fb, acc_rf = [], []
        cur = node
        while cur is not None:
            if cur["test_feedback"]:
                acc_fb.append(cur["test_feedback"])
            if cur["reflection"]:
                acc_rf.append(cur["reflection"])
            cur = cur["pref"]
        acc_fb, acc_rf = acc_fb[::-1], acc_rf[::-1]   # 官方反转：最早在前
        it_rec = {"iter": cur_iter, "selected_path": path,
                  "new_ids": new_ids, "simulated": simulated,
                  "reward_internal": reward_internal,
                  "reward_real": reward_real,
                  "updated_child": c["id"], "backprop": [],
                  "solved": is_solved, "acc_feedback": acc_fb,
                  "acc_reflection": acc_rf,
                  "update_child_reflection": c["reflection"]}
        reward = reward_internal + reward_real
        c["visits"] += 1
        c["value"] += reward
        temp = c
        while temp["pref"] is not None:
            temp = temp["pref"]
            temp["visits"] += 1
            temp["value"] += reward
            it_rec["backprop"].append(temp["id"])
        iters.append(it_rec)

    if is_solved:
        best = solved_child
    else:
        root = nodes[0]
        if not root["children"]:
            raise ValueError("no children")
        best = None
        for c in root["children"]:              # max by value 平分取先
            if best is None or c["value"] > best["value"]:
                best = c
    final = {"best_solution": best["solution"], "best_child_id": best["id"],
             "solved": is_solved}

    def clean(nd):
        return {k: (v if k not in ("children",) else
                    [clean(c) for c in v])
                for k, v in nd.items() if k != "pref"}

    out_nodes = [clean(nd) for nd in nodes]
    return {"iters": iters, "nodes": out_nodes, "final": final}


# ---------------------------------------------------------------- compare

def _eq(a, b):
    if isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), (len(a), len(b))
        for x, y in zip(a, b):
            if isinstance(x, (list, dict)):
                _eq(x, y)
            elif isinstance(x, float) and isinstance(y, float):
                assert math.isclose(x, y, rel_tol=0, abs_tol=1e-12), (x, y)
            else:
                assert x == y, (x, y)
        return
    assert isinstance(a, dict) and isinstance(b, dict), (a, b)
    assert a.keys() == b.keys(), (a.keys() ^ b.keys())
    for k in a:
        if isinstance(a[k], float) and isinstance(b[k], float):
            assert math.isclose(a[k], b[k], rel_tol=0, abs_tol=1e-12), \
                (k, a[k], b[k])
        elif isinstance(a[k], (list, dict)):
            _eq(a[k], b[k])
        else:
            assert a[k] == b[k], (k, a[k], b[k])


def _compare(solve, *args, **kw):
    got = solve(_dc(args[0]), *[_dc(x) for x in args[1:]],
                **{k: _dc(v) for k, v in kw.items()})
    exp = _oracle(_dc(args[0]), *[_dc(x) for x in args[1:]],
                  **{k: _dc(v) for k, v in kw.items()})
    _eq(got, exp)
    return got


# ---------------------------------------------------------------- checks

def check_single_iter_all_fail(solve):
    """单迭代 n=3 全失败：只有循环变量泄漏的最后一个 child 得 reward；
    reward_internal 来自最后 child 的 feedback 计数。"""
    plan = {"solutions": ["s1", "s2", "s3"],
            "sims": [(False, "no output", "r1", None),
                     (False, "no output", "r2", None),
                     (False, "Tested passed:\nt1\nt2\nTests failed:\nt3",
                      "r3", None)]}
    led = _compare(solve, "root", 2, 3, 4, [plan])
    it0 = led["iters"][0]
    assert it0["new_ids"] == [1, 2, 3]
    assert it0["simulated"] == [1, 2, 3]
    assert it0["updated_child"] == 3            # 泄漏的最后一个
    # "Tested passed:" 后非空行 2 行 / n_tests 4
    assert math.isclose(it0["reward_internal"], 2 / 4, abs_tol=1e-12)
    assert it0["reward_real"] == 0
    assert it0["backprop"] == [0]               # root
    n3 = led["nodes"][3]
    assert n3["visits"] == 1 and math.isclose(n3["value"], 0.5,
                                              abs_tol=1e-12)
    n1 = led["nodes"][1]
    assert n1["visits"] == 0 and n1["value"] == 0   # 未被 update
    assert n1["reflection"] == "r1"
    assert led["final"]["solved"] is False
    assert led["final"]["best_child_id"] == 3   # value 0.5 最大


def check_uct_zero_visits_value(solve):
    """UCT visits==0 返回 value（官方注释掉 inf）；parent.visits=1 时
    探索项 ln(1)=0 → 已访问子节点 uct=均值。"""
    # iter1: n=2, c2 得 reward 1.0，c1 未访问（uct=0）
    plan1 = {"solutions": ["a", "b"],
             "sims": [(False, "err", "ra", None),
                      (True, "Tested passed:\nok", "", False)]}
    # c2 passing → evaluate False（ev=False）→ break，reward_internal=1
    led = _compare(solve, "root", 2, 2, 1, [plan1])
    it0 = led["iters"][0]
    assert it0["updated_child"] == 2
    assert math.isclose(it0["reward_internal"], 1.0, abs_tol=1e-12)
    # iter2 selection：c1 uct=0（visits==0→value），c2 uct=1/1+sqrt(ln(1)/1)=1
    # → 选 c2（value 更大）
    plan2 = {"solutions": ["c", "d"],
             "sims": [(False, "err", "rc", None),
                      (False, "err", "rd", None)]}
    led2 = _compare(solve, "root", 2, 2, 1, [plan1, plan2])
    assert led2["iters"][1]["selected_path"] == [0, 2]
    # c2 的 children 3/4；root.value=1+... 检查 backprop
    assert led2["iters"][1]["backprop"] == [2, 0]


def check_uct_tie_first(solve):
    """UCT 平分取先扩展者：两 child 都未访问且 value 同为 0 → 取先。"""
    plan1 = {"solutions": ["a", "b"],
             "sims": [(False, "err", "ra", None),
                      (False, "err", "rb", None)]}
    # 全失败 reward=0：c1/c2 均 visits=0 value=0 → uct 平分 → 先者 c1
    plan2 = {"solutions": ["c"], "sims": [(False, "err", "rc", None)]}
    led = _compare(solve, "root", 2, 2, 1, [plan1, plan2])
    assert led["iters"][1]["selected_path"] == [0, 1]


def check_break_on_old_child(solve):
    """selection 恒下行到叶子 → 内层模拟的 children 恰为当轮新加的
    n 个（官方 for child in node.children 不会碰到历史 child）；
    break 时正在处理的 child 是 update 对象。"""
    plan1 = {"solutions": ["a", "b"],
             "sims": [(False, "err", "ra", None),
                      (False, "err", "rb", None)]}
    # iter2：selection 选 c1（uct 平分取先）→ 叶子；新 child 3；
    # c3 passing → evaluate False → break → c3 是 update 对象
    plan2 = {"solutions": ["c"],
             "sims": [(True, "Tested passed:\nok", "ra2", False),
                      (False, "err", "rc", None)]}
    plan3 = {"solutions": ["z"],
             "sims": [(False, "err", "rz", None)]}
    led = _compare(solve, "root", 3, 2, 1, [plan1, plan2, plan3])
    it1 = led["iters"][1]
    assert it1["selected_path"] == [0, 1]
    assert it1["new_ids"] == [3]
    assert it1["simulated"] == [3]              # 只有新 child 被模拟
    assert it1["updated_child"] == 3            # break 处的 child
    assert it1["reward_real"] == 0              # eval False
    # c3.value = reward_internal(1.0) + 0；iter2 末轮又 backprop +1 访问
    assert math.isclose(led["nodes"][3]["value"], 1.0, abs_tol=1e-12)
    assert led["nodes"][3]["visits"] == 2
    assert it1["backprop"] == [1, 0]
    # 历史 c1 的反思/反馈不被触碰（selection 端点是叶子）
    assert led["nodes"][1]["reflection"] == "ra"
    assert led["nodes"][1]["test_feedback"] == "err"


def check_solve_exit(solve):
    """eval 通过 → solved：reward_real=1、提前退出全部迭代、
    best=触发 solve 的 child。"""
    plan1 = {"solutions": ["a", "b"],
             "sims": [(True, "Tested passed:\nok", "", True)]}
    led = _compare(solve, "root", 5, 2, 1, [plan1])
    assert led["final"]["solved"] is True
    assert led["final"]["best_solution"] == "a"
    assert led["final"]["best_child_id"] == 1
    assert len(led["iters"]) == 1               # 提前退出
    it0 = led["iters"][0]
    assert it0["reward_real"] == 1
    assert it0["solved"] is True
    # reward = 1.0 + 1 = 2.0
    assert math.isclose(led["nodes"][1]["value"], 2.0, abs_tol=1e-12)


def check_last_iter_eval_first_child(solve):
    """最后一轮：cur_iter==max_iters-1 对**第一个** child 即 evaluate
    并 break——后续 child 不再模拟。"""
    plan = {"solutions": ["a", "b", "c"],
            "sims": [(False, "err", "ra", False),
                     (False, "err", "rb", False),
                     (False, "err", "rc", False)]}
    led = _compare(solve, "root", 1, 3, 1, [plan])
    it0 = led["iters"][0]
    assert it0["simulated"] == [1]              # 只模拟第一个
    assert it0["updated_child"] == 1
    assert it0["reward_real"] == 0              # eval False


def check_gather_context(solve):
    """gather_context_from_tree：对选择端点（父）收集、最早在前、
    含 root 的首轮 feedback/reflection。"""
    plan1 = {"solutions": ["a"],
             "sims": [(False, "fb1", "refl1", None)]}
    led = _compare(solve, "root", 2, 1, 1, [plan1],
                   root_feedback="root_fb", root_reflection="root_refl")
    # iter1 选择端点是 root（无 children）：acc = [root_fb], [root_refl]
    it0 = led["iters"][0]
    assert it0["acc_feedback"] == ["root_fb"]
    assert it0["acc_reflection"] == ["root_refl"]
    # iter2 选择端点是 c1（visits=0 → uct=value=0.5？c1 有 reward）
    plan2 = {"solutions": ["b"],
             "sims": [(False, "fb2", "refl2", None)]}
    led2 = _compare(solve, "root", 2, 1, 1, [plan1, plan2],
                    root_feedback="root_fb", root_reflection="root_refl")
    it1 = led2["iters"][1]
    # c1.test_feedback=fb1, c1.reflection=refl1；root 有 root_fb/root_refl
    assert it1["acc_feedback"] == ["root_fb", "fb1"]
    assert it1["acc_reflection"] == ["root_refl", "refl1"]
    assert it1["update_child_reflection"] == "refl2"


def check_best_child_value_tie(solve):
    """终局 best_child_value：max by 累计 value，平分取先。"""
    plan1 = {"solutions": ["a", "b"],
             "sims": [(False, "err", "ra", None),
                      (False, "err", "rb", None)]}
    led = _compare(solve, "root", 1, 2, 1, [plan1])
    # 全零 value → 取先扩展的 c1
    assert led["final"]["best_child_id"] == 1
    assert led["final"]["best_solution"] == "a"
    # 让 c2 得正 reward：n=2 只 update 最后 child=c2
    plan2 = {"solutions": ["a", "b"],
             "sims": [(False, "err", "ra", None),
                      (True, "Tested passed:\nok", "", False)]}
    plan2b = {"solutions": ["x", "y"],
              "sims": [(False, "err", "rx", None),
                       (False, "err", "ry", None)]}
    led2 = _compare(solve, "root", 2, 2, 1, [plan2, plan2b])
    assert led2["final"]["best_child_id"] == 2
    assert led2["final"]["best_solution"] == "b"


def check_domain_errors(solve):
    """solved 未发生且 root 无 children → ValueError；expansion 零
    产出 → ValueError。"""
    try:
        solve("root", 0, 2, 1, [])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError (no children)")
    try:
        solve("root", 1, 2, 1, [])      # plans 空 → 提前结束 → 无 children
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError (empty plans)")
    try:
        solve("root", 1, 2, 1,
              [{"solutions": [], "sims": []}])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError (no expansion)")


def check_reward_counting(solve):
    """reward_internal 计数细节：'Tested passed:' 拼写、'Tests failed:'
    截断、空行不计、无标记则 0。"""
    fb = "Tested passed:\nt1\n\nt2\nTests failed:\nt3\n"
    plan = {"solutions": ["s"],
            "sims": [(False, fb, "r", None)]}
    led = _compare(solve, "root", 1, 1, 3, [plan])
    # 非空行 t1/t2 → 2/3
    assert math.isclose(led["iters"][0]["reward_internal"], 2 / 3,
                        abs_tol=1e-12)
    # 无 "Tested passed:" → 0
    plan2 = {"solutions": ["s"], "sims": [(False, "all bad", "r", None)]}
    led2 = _compare(solve, "root", 1, 1, 3, [plan2])
    assert led2["iters"][0]["reward_internal"] == 0
    # 无 "Tests failed:" 段 → 整段计数
    fb3 = "Tested passed:\nx\ny\nz"
    plan3 = {"solutions": ["s"], "sims": [(False, fb3, "r", None)]}
    led3 = _compare(solve, "root", 1, 1, 4, [plan3])
    assert math.isclose(led3["iters"][0]["reward_internal"], 3 / 4,
                        abs_tol=1e-12)
    # 纯空格行不算非空行
    fb4 = "Tested passed:\nt1\n \nt2\nTests failed:\n"
    plan4 = {"solutions": ["s"], "sims": [(False, fb4, "r", None)]}
    led4 = _compare(solve, "root", 1, 1, 4, [plan4])
    assert math.isclose(led4["iters"][0]["reward_internal"], 2 / 4,
                        abs_tol=1e-12)


def check_uct_backprop_accumulation(solve):
    """backprop 累积账：c1 直接 update + 经 c3 的 backprop 各一次
    （visits=2、value=2.0）；c3 直接 update（visits=1、value=1.0）。
    iter2 selection：root 层 c1(1.59) 胜 c2(0)，c1 层 c3(1.83) 胜
    c4(0)——uct 用各自父的 visits。"""
    ok = "Tested passed:\nok"
    plans = [
        {"solutions": ["a", "b"],
         "sims": [(True, ok, "", False), (False, "err", "rb", None)]},
        {"solutions": ["c", "d"],
         "sims": [(True, ok, "", False), (False, "err", "rd", None)]},
        {"solutions": ["e", "f"],
         "sims": [(False, "err", "re", None), (False, "err", "rf", None)]},
        {"solutions": ["g", "h"],
         "sims": [(False, "err", "rg", None), (False, "err", "rh", None)]},
    ]
    led = _compare(solve, "root", 4, 2, 1, plans)
    # iter2 selection：root 层选 c1，c1 层选 c3（各用其父 visits）
    assert led["iters"][2]["selected_path"] == [0, 1, 3]
    assert led["iters"][2]["new_ids"] == [5, 6]
    # 终态：iter2/iter3 的 backprop 继续累积（reward 0 只加 visits）
    n1 = led["nodes"][1]
    assert n1["visits"] == 4 and math.isclose(n1["value"], 2.0,
                                              abs_tol=1e-12)
    n3 = led["nodes"][3]
    assert n3["visits"] == 3 and math.isclose(n3["value"], 1.0,
                                              abs_tol=1e-12)


def check_random(solve):
    """120 组随机 MCTS 运行对拍 oracle。"""
    rng = random.Random(20261102)

    def fb_gen():
        if rng.random() < 0.4:
            k = rng.randint(1, 3)
            lines = "".join(f"t{i}\n" for i in range(k))
            f = rng.randint(0, 2)
            fl = "".join(f"x{i}\n" for i in range(f))
            return f"Tested passed:\n{lines}Tests failed:\n{fl}"
        return rng.choice(["err", "syntax", "ok maybe"])

    for _ in range(120):
        iters = rng.randint(1, 3)
        n = rng.randint(1, 3)
        n_tests = rng.randint(1, 4)
        plans = []
        for _it in range(iters):
            nsol = rng.randint(0, n)
            plans.append({"solutions": [f"s{rng.randint(0, 9)}"
                                        for _ in range(nsol)],
                          "sims": [(rng.random() < 0.3, fb_gen(),
                                    f"r{rng.randint(0, 99)}",
                                    rng.random() < 0.3)
                                   for _ in range(24)]})
        args = ("root", iters, n, n_tests, plans)
        try:
            got = solve(*_dc(args))
            err = None
        except Exception as e:
            got, err = None, type(e).__name__
        try:
            exp = _oracle(*_dc(args))
            exp_err = None
        except Exception as e:
            exp, exp_err = None, type(e).__name__
        if err is not None or exp_err is not None:
            assert err == exp_err, (err, exp_err)
            continue
        _eq(got, exp)
