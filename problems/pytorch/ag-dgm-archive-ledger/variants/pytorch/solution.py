"""Darwin Gödel Machine 存档进化：父代选择、条目分派与入档判定账。

对照 jennyzzt/dgm 的 `DGM_outer.py` 与 `utils/evo_utils.py`：
- `choose_selfimproves` 建候选表（含 `children_count`）、按 sigmoid 分数与子代数量给父代概率、
  再为每个父代顺序过三道 25% 闸门挑自改条目；
- `filter_compiled` 用 `is_compiled_self_improve` 刷掉没编译 / 全空补丁 / 题数不足的子代；
- `update_archive` 的 `keep_better` 用「初始分 - 噪声余量」当门槛；
- `get_full_eval_threshold` 取「初始分 + 满评存档分」降序第二高，再与 0.4 取大。

随机段（父代抽样、闸门抛硬币、条目抽取）全部改成显式入参。
"""
import math

_METHODS = ("random", "score_prop", "score_child_prop", "best")
_UPDATES = ("keep_all", "keep_better")
_GATES = ("solve_empty_patches", "solve_stochasticity", "solve_contextlength")


def _as_int(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("%s 必须是整数" % name)
    if minimum is not None and value < minimum:
        raise ValueError("%s 必须 >= %d" % (name, minimum))
    return value


def _as_real(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s 必须是实数" % name)
    out = float(value)
    if not math.isfinite(out):
        raise ValueError("%s 不得为 nan / inf" % name)
    return out


def _as_commit(raw, name):
    if not isinstance(raw, dict):
        raise ValueError("%s 必须是字典" % name)
    for key in ("id", "score", "resolved", "unresolved", "empty", "submitted"):
        if key not in raw:
            raise ValueError("%s 缺少字段 %r" % (name, key))
    if not isinstance(raw["id"], str) or not raw["id"]:
        raise ValueError("%s.id 必须是非空字符串" % name)
    out = {"id": raw["id"], "score": _as_real(raw["score"], "%s.score" % name),
           "resolved": _as_int(raw["resolved"], "%s.resolved" % name, 0),
           "unresolved": _as_int(raw["unresolved"], "%s.unresolved" % name, 0),
           "empty": _as_int(raw["empty"], "%s.empty" % name, 0),
           "submitted": _as_int(raw["submitted"], "%s.submitted" % name, 0)}
    parent = raw.get("parent")
    if parent is not None and (not isinstance(parent, str) or not parent):
        raise ValueError("%s.parent 必须是非空字符串或 None" % name)
    out["parent"] = parent
    out["metadata_ok"] = bool(raw.get("metadata_ok", True))
    out["context_exceeded"] = bool(raw.get("context_exceeded", False))
    return out


def _sigmoid(score):
    """`1 / (1 + exp(-10 (s - 0.5)))`，按 z 的符号拆两支避免 exp 溢出。"""
    z = 10.0 * (score - 0.5)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def solve(commits, children, method="score_child_prop", size=2,
          update="keep_all", noise_leeway=0.1, num_issues=60,
          num_full_eval=500, parent_picks=None, pad_picks=None, draws=None,
          baseline=None):
    if not isinstance(commits, (list, tuple)) or not commits:
        raise ValueError("commits 必须是非空序列")
    if not isinstance(children, (list, tuple)):
        raise ValueError("children 必须是序列")
    per_gen = _as_int(size, "size", 1)
    if method not in _METHODS:
        raise ValueError("未知的 method: %r" % (method,))
    if update not in _UPDATES:
        raise ValueError("未知的 update: %r" % (update,))
    leeway = _as_real(noise_leeway, "noise_leeway")
    issues = _as_int(num_issues, "num_issues", 0)
    full_eval = _as_int(num_full_eval, "num_full_eval", 1)
    if baseline not in (None, "no_darwin", "no_selfimprove"):
        raise ValueError("未知的 baseline: %r" % (baseline,))

    archive = [_as_commit(raw, "commits[%d]" % i) for i, raw in enumerate(commits)]
    new_ids = [_as_commit(raw, "children[%d]" % i) for i, raw in enumerate(children)]
    by_id = {c["id"]: c for c in archive}
    if "initial" not in by_id:
        raise ValueError("存档里必须有 initial")
    if len(by_id) != len(archive):
        raise ValueError("存档内 id 重复")
    if len({c["id"] for c in new_ids}) != len(new_ids):
        raise ValueError("children 内 id 重复")

    # ---- 候选表：metadata 读不出来就不是候选；父代不在候选表里则不计数 ----
    candidates, order = {}, []
    for commit in archive:
        if not commit["metadata_ok"]:
            continue
        candidates[commit["id"]] = {
            "id": commit["id"], "score": commit["score"],
            "resolved": commit["resolved"], "unresolved": commit["unresolved"],
            "empty": commit["empty"], "submitted": commit["submitted"],
            "context_exceeded": commit["context_exceeded"],
            "children_count": 0,
        }
        order.append(commit["id"])
    for commit in archive:
        if commit["id"] not in candidates or commit["parent"] is None:
            continue
        if commit["parent"] in candidates:
            candidates[commit["parent"]]["children_count"] += 1

    # ---- 父代选择 ----
    probabilities = None
    if baseline == "no_darwin":
        parents = order[-1:]
        if parent_picks is not None:
            raise ValueError("no_darwin 下不接受 parent_picks")
    elif method in ("score_prop", "score_child_prop"):
        raw = [_sigmoid(candidates[cid]["score"]) for cid in order]
        if method == "score_child_prop":
            raw = [s / (1 + candidates[cid]["children_count"])
                   for s, cid in zip(raw, order)]
        total = sum(raw)
        probabilities = [x / total for x in raw]
        if not isinstance(parent_picks, (list, tuple)):
            raise ValueError("需要显式的 parent_picks")
        if len(parent_picks) != per_gen:
            raise ValueError("parent_picks 数量必须是 size")
        parents = []
        for raw_idx in parent_picks:
            idx = _as_int(raw_idx, "parent_picks 项", 0)
            if idx >= len(order):
                raise ValueError("parent_picks 越界")
            parents.append(order[idx])
    elif method == "best":
        # 官方实现按分数**升序**排后取前 size 个——照抄：这等于取最差的那几个
        ranked = sorted(order, key=lambda cid: candidates[cid]["score"])
        parents = ranked[:min(per_gen, len(ranked))]
        if parent_picks is not None:
            raise ValueError("best 下不接受 parent_picks")
        if len(parents) < per_gen:
            need = per_gen - len(parents)
            if not isinstance(pad_picks, (list, tuple)) or len(pad_picks) != need:
                raise ValueError("best 名额不足时需要 %d 个 pad_picks" % need)
            for raw_idx in pad_picks:
                idx = _as_int(raw_idx, "pad_picks 项", 0)
                if idx >= len(parents):
                    raise ValueError("pad_picks 越界")
                parents.append(parents[idx])
    else:
        if not isinstance(parent_picks, (list, tuple)):
            raise ValueError("需要显式的 parent_picks")
        if len(parent_picks) != per_gen:
            raise ValueError("parent_picks 数量必须是 size")
        parents = []
        for raw_idx in parent_picks:
            idx = _as_int(raw_idx, "parent_picks 项", 0)
            if idx >= len(order):
                raise ValueError("parent_picks 越界")
            parents.append(order[idx])

    # ---- 条目分派：三道 25% 闸门按序短路，未命中的抛硬币不消耗 ----
    if not isinstance(draws, (list, tuple)):
        raise ValueError("draws 必须是序列")
    stream = [_as_real(u, "draws 项") for u in draws]
    for u in stream:
        if u < 0.0 or u >= 1.0:
            raise ValueError("draws 每项必须落在 [0, 1)")
    cursor = 0

    def take(name):
        nonlocal cursor
        if cursor >= len(stream):
            raise ValueError("draws 不够用（%s）" % name)
        cursor += 1
        return stream[cursor - 1]

    entries = []
    for parent in parents:
        cand = candidates[parent]
        total_ids = cand["empty"] + cand["resolved"] + cand["unresolved"]
        if cand["empty"] >= 0.1 * total_ids:      # 第一道：空补丁占比，短路不抛硬币
            if take("empty_patches") < 0.25:
                entries.append({"parent": parent, "entry": "solve_empty_patches",
                                "index": None})
                continue
        if take("stochasticity") < 0.25:
            entries.append({"parent": parent, "entry": "solve_stochasticity",
                            "index": None})
            continue
        if cand["context_exceeded"]:              # 第三道：只有超长才抛硬币
            if take("contextlength") < 0.25:
                entries.append({"parent": parent, "entry": "solve_contextlength",
                                "index": None})
                continue
        if cand["unresolved"] == 0:               # 没未解决项：直接跳过，不再抽
            entries.append({"parent": parent, "entry": "skipped", "index": None})
            continue
        pick = take("unresolved")
        index = min(int(pick * cand["unresolved"]), cand["unresolved"] - 1)
        entries.append({"parent": parent, "entry": "unresolved", "index": index})

    # ---- 编译过滤：三条件全过才算子代 ----
    compiled, rejected = [], []
    for child in new_ids:
        if not child["metadata_ok"]:
            rejected.append({"id": child["id"], "reason": "no_metadata"})
        elif child["resolved"] + child["unresolved"] == 0:
            rejected.append({"id": child["id"], "reason": "no_patch"})
        elif child["submitted"] < issues:
            rejected.append({"id": child["id"], "reason": "not_enough_tests"})
        else:
            compiled.append(child["id"])

    # ---- 满评门槛：初始分必进，满评存档成员再按 submitted 过筛（initial 会被数两次）----
    pool = [by_id["initial"]["score"]]
    for commit in archive:
        if commit["submitted"] < full_eval * 0.9:
            continue
        pool.append(commit["score"])
    ranked_scores = sorted(pool, reverse=True)
    threshold = ranked_scores[1] if len(ranked_scores) > 1 else ranked_scores[0]
    threshold = max(threshold, 0.4)

    # ---- 入档 ----
    after = [c["id"] for c in archive]
    gate = by_id["initial"]["score"] - leeway
    if update == "keep_better":
        for child in new_ids:
            if child["id"] in compiled and child["score"] >= gate:
                after.append(child["id"])
    else:
        after.extend(compiled)

    return {
        "candidates": [candidates[cid] for cid in order],
        "probabilities": probabilities,
        "parents": parents,
        "entries": entries,
        "draws_used": cursor,
        "compiled": compiled,
        "rejected": rejected,
        "archive": after,
        "threshold": threshold,
    }
