"""
Graph of Thoughts (GoT) 操作图账：把 spcl/graph-of-thoughts 的
operations.py（Thought + Operation 家族）在**线性操作链**上无状态化。

输入 ops 为操作描述符列表，按 append_operation 顺序组成一条链：
每个操作从唯一前驱取 thoughts（首个操作无前驱）。

操作描述符：
- {"op": "score", "combined": bool, "scores": [...]}
    combined=True：把全部前驱 thought 的 state 一次性交给打分，scores
    是与 thoughts zip 的列表——zip 截断：scores 更短时多出的 thought
    被静默丢弃。combined=False：逐 thought 打分，scores 逐条消费。
    打分走 score setter（scored=True）。
- {"op": "validate_and_improve", "improve": bool, "num_tries": int,
   "valids": [...], "updates": [...]}
    每次验证从 valids 消费一个布尔（写入 valid setter → validated=True），
    当前 attempt 追加进 attempt 链；若 improve 且无效且尝试次数
    current_try < num_tries：从 updates 消费一个 dict 做
    {**state, **update} 合并进入下一轮，current_try += 1。
    恒定无效时验证恰 num_tries+1 次。最终 thought 取 attempt 链末位。
- {"op": "generate", "num_branches_prompt": p, "num_branches_response": r,
   "parsed": [[state, ...], ...]}
    前驱为空且有前驱 → 本操作 thoughts 置空（直接返回）；
    parsed 逐父槽位消费，每个新 state 以 {**父state, **new_state} 合并。
    产物数量超出 p*r*父数只是官方日志警告，**不截断**。
- {"op": "improve", "updates": [...]}：逐 thought 消费 update，
    {**state, **update} 生成新 thought。
- {"op": "aggregate", "parsed": dict 或 [dict, ...]}
    base_state 按前驱 thoughts **score 升序**（stable）逐个
    {**base, **state} 合并——score 高的后合并、覆盖同名键；
    未打分 thought 用缺省 score=0.0，平分保持原序。
    parsed 若为 dict 包装成单元素列表，逐个 {**base_state, **p} 入列。
- {"op": "keep_best_n", "n": int, "higher_is_better": bool}
    全部前驱必须已打分，否则 ValueError；按 score 排序（stable，
    平分保持原下标序）取前 n。排序失败（如含 score=None 但已打分）
    时官方 fallback：只保留 isinstance(score, float) 的 thought
    再排序取前 n——int score 也会被丢弃。
    n<=0 构造期即 ValueError。
- {"op": "keep_valid"}：保留 `not validated or valid` 的 thought
    ——**未验证的 thought 直接放行**，只有"已验证且无效"才被剔除。
- {"op": "ground_truth", "results": [...], "raises": [...]}
    逐 thought 消费：raises[i] 为真则求值抛异常 → solved=False；
    否则 solved=results[i]。solved setter 总置
    compared_to_ground_truth=True（结果为 False 也置）。
- {"op": "selector", "indices": [...]}：按索引选取前驱 thoughts。

每个 thought 记账为 {"state", "score", "scored", "valid", "validated",
"solved", "compared_to_ground_truth"}；Thought 缺省 score=0.0、
valid=False、solved=False，三个布尔标志缺省 False。
from_thought 克隆会复制全部标志（state 为共享引用）。

solve 返回 {"steps": [每操作记录], "final": 最后一个操作的 thoughts}。
每条记录含 {"op", "count", "thoughts"} 及操作特有键：
- score：{"scores": 本操作实际赋值列表}
- validate_and_improve：{"attempts": 每输入 thought 的 attempt 数}
- generate：{"skipped": 是否因前驱为空而跳过}
- aggregate：{"base_state": 合并出的基态}
- keep_best_n：{"fallback": 是否走了 float-only fallback}
- ground_truth：{"solved_count"}

域约束（ValueError）：keep_best_n 的 n<=0、存在未打分 thought；
aggregate/keep_best_n/keep_valid/ground_truth 在无前驱时 ValueError
（对应官方 assert，转 ValueError）。score/validate_and_improve/
improve 无前驱同理。

函数签名：solve(ops, base_state=None)
- 首个操作为 generate/selector 且 base_state 给出时，用作
  Thought(kwargs) 的种子态。
"""
from copy import deepcopy


def _fresh_thought(state, score=0.0, scored=False, valid=False,
                   validated=False, solved=False, compared=False):
    # 内部 thought 表示：dict 携带全部官方标志
    return {
        "state": state,
        "score": score,
        "scored": scored,
        "valid": valid,
        "validated": validated,
        "solved": solved,
        "compared_to_ground_truth": compared,
    }


def _from_thought(t):
    # Thought.from_thought：复制全部标志（state 共享引用——
    # 账面只读，统一 deepcopy 防止跨步 aliasing 影响断言）
    return _fresh_thought(deepcopy(t["state"]), t["score"], t["scored"],
                          t["valid"], t["validated"], t["solved"],
                          t["compared_to_ground_truth"])


def _view(t):
    return {
        "state": t["state"],
        "score": t["score"],
        "scored": t["scored"],
        "valid": t["valid"],
        "validated": t["validated"],
        "solved": t["solved"],
        "compared_to_ground_truth": t["compared_to_ground_truth"],
    }


def _record(op, thoughts, **extra):
    rec = {"op": op, "count": len(thoughts),
           "thoughts": [_view(t) for t in thoughts]}
    rec.update(extra)
    return rec


def solve(ops, base_state=None):
    prev = []
    steps = []
    first = True

    for spec in ops:
        kind = spec["op"]

        if kind == "score":
            if first:
                raise ValueError("Score operation needs at least one predecessor")
            scores = list(spec["scores"])
            assigned = []
            if spec.get("combined"):
                # combined：zip 截断——scores 短则多出的 thought 丢弃
                new_prev = []
                for t, s in zip(prev, scores):
                    nt = _from_thought(t)
                    nt["score"] = s
                    nt["scored"] = True
                    assigned.append(s)
                    new_prev.append(nt)
                prev = new_prev
            else:
                new_prev = []
                for i, t in enumerate(prev):
                    s = scores[i]
                    nt = _from_thought(t)
                    nt["score"] = s
                    nt["scored"] = True
                    assigned.append(s)
                    new_prev.append(nt)
                prev = new_prev
            steps.append(_record("score", prev, scores=assigned))

        elif kind == "validate_and_improve":
            if first:
                raise ValueError(
                    "ValidateAndImprove operation needs at least one predecessor")
            valids = list(spec["valids"])
            updates = list(spec.get("updates", []))
            improve = spec.get("improve", True)
            num_tries = spec.get("num_tries", 3)
            attempts = []
            out = []
            vi = 0
            ui = 0
            for t in prev:
                chain = []
                cur = _from_thought(t)
                cur_try = 0
                while True:
                    valid = valids[vi]
                    vi += 1
                    cur["valid"] = valid
                    cur["validated"] = True
                    chain.append(cur)
                    if not improve or valid or cur_try >= num_tries:
                        break
                    upd = updates[ui]
                    ui += 1
                    # 官方 Thought({**state, **update})：全新缺省
                    # thought——score 复位 0.0、solved 等标志全复位
                    cur = _fresh_thought({**cur["state"], **upd})
                    cur_try += 1
                attempts.append(len(chain))
                out.append(chain[-1])
            prev = out
            steps.append(_record("validate_and_improve", prev,
                                 attempts=attempts))

        elif kind == "generate":
            if not prev and not first:
                # 官方：有前驱但前驱无 thoughts → 直接返回空
                steps.append(_record("generate", [], skipped=True))
                prev = []
                first = False
                continue
            parents = prev if prev else \
                [_fresh_thought(deepcopy(base_state) if base_state else {})]
            p = spec.get("num_branches_prompt", 1)
            r = spec.get("num_branches_response", 1)
            parsed = spec["parsed"]
            out = []
            for t, new_states in zip(parents, parsed):
                for ns in new_states:
                    out.append(_fresh_thought({**t["state"], **ns}))
            prev = out
            steps.append(_record("generate", prev, skipped=False))

        elif kind == "improve":
            if first:
                raise ValueError("Needs at least one predecessor")
            updates = list(spec["updates"])
            out = []
            # 官方 Thought({**state, **update})：全新缺省 thought
            # ——score 复位 0.0、valid/solved 等标志全复位
            for t, upd in zip(prev, updates):
                out.append(_fresh_thought({**t["state"], **upd}))
            prev = out
            steps.append(_record("improve", prev))

        elif kind == "aggregate":
            if first:
                raise ValueError(
                    "Aggregate operation must have at least one predecessor")
            if len(prev) == 0:
                # 官方：有前驱但零 thoughts → 直接返回空（不报错）
                steps.append(_record("aggregate", []))
                first = False
                continue
            # applied in order of score：score 升序 stable 合并，
            # 高 score 后合并、覆盖同名键；未打分缺省 0.0
            base = {}
            for t in sorted(prev, key=lambda x: x["score"]):
                base = {**base, **t["state"]}
            parsed = spec["parsed"]
            if isinstance(parsed, dict):
                parsed = [parsed]
            out = []
            for ns in parsed:
                out.append(_fresh_thought({**base, **ns}))
            prev = out
            steps.append(_record("aggregate", prev, base_state=deepcopy(base)))

        elif kind == "keep_best_n":
            if first:
                raise ValueError(
                    "KeepBestN operation must have at least one predecessor")
            n = spec["n"]
            if n <= 0:
                raise ValueError(
                    "KeepBestN operation must keep at least one thought")
            higher = spec.get("higher_is_better", True)
            if not all(t["scored"] for t in prev):
                raise ValueError("Not all thoughts have been scored")
            fallback = False
            try:
                ordered = sorted(prev, key=lambda t: t["score"],
                                 reverse=higher)
            except TypeError:
                fallback = True
                ordered = sorted(
                    [t for t in prev if isinstance(t["score"], float)],
                    key=lambda t: t["score"], reverse=higher)
            kept = [_from_thought(t) for t in ordered[:n]]
            prev = kept
            steps.append(_record("keep_best_n", prev, fallback=fallback))

        elif kind == "keep_valid":
            if first:
                raise ValueError(
                    "KeepValid operation must have at least one predecessor")
            kept = [_from_thought(t) for t in prev
                    if not t["validated"] or t["valid"]]
            prev = kept
            steps.append(_record("keep_valid", prev))

        elif kind == "ground_truth":
            if first:
                raise ValueError(
                    "GroundTruth operation must have at least one predecessor")
            results = list(spec["results"])
            raises = list(spec.get("raises", [False] * len(results)))
            out = []
            solved_count = 0
            for i, t in enumerate(prev):
                nt = _from_thought(t)
                if raises[i]:
                    nt["solved"] = False
                else:
                    nt["solved"] = results[i]
                nt["compared_to_ground_truth"] = True
                if nt["solved"]:
                    solved_count += 1
                out.append(nt)
            prev = out
            steps.append(_record("ground_truth", prev,
                                 solved_count=solved_count))

        elif kind == "selector":
            if not prev and first:
                prev = [_fresh_thought(deepcopy(base_state)
                                       if base_state else {})]
            idx = spec["indices"]
            prev = [_from_thought(prev[i]) for i in idx]
            steps.append(_record("selector", prev))

        else:
            raise ValueError(f"unknown operation: {kind}")

        first = False

    return {"steps": steps, "final": [_view(t) for t in prev]}
