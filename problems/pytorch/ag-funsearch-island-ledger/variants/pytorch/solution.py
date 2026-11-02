"""FunSearch 岛屿程序库：聚类注册、温度调度采样与弱岛重置账。

对照 google-deepmind/funsearch `implementation/programs_database.py`：
程序按逐测试分数向量聚成簇（Cluster），簇的分数只在**首个**成员进簇时定下；
取样时的簇概率是分数 / 温度的 softmax，温度随该岛累计程序数线性衰减；
弱岛重置时把最优程序克隆过去当创始人。
随机段（选岛、簇内取样、重置噪声、创始人抽取）全部改成显式入参。
"""
import math

_KINDS = ("register", "prompt", "reset")


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


def _softmax(logits, temperature):
    """按 1D 有限 logits 求高温 / 低温 softmax（减最大值保数值稳定）。"""
    if not logits:  # 空岛不在这里兜底，由调用处显式报错
        return []
    scaled = [x / temperature for x in logits]
    peak = max(scaled)
    exps = [math.exp(x - peak) for x in scaled]
    total = sum(exps)
    return [x / total for x in exps]


def _signature(scores):
    """逐测试分数按**键名升序**拼成簇键。"""
    return tuple(scores[k] for k in sorted(scores))


def _reduce(scores):
    """簇分取**最后插入**那个测试的分数（不是最大、也不是字典序末位）。"""
    return scores[list(scores)[-1]]


def _as_program(program, name):
    if not isinstance(program, dict):
        raise ValueError("%s 必须是字典" % name)
    for key in ("id", "doc", "size"):
        if key not in program:
            raise ValueError("%s 缺少字段 %r" % (name, key))
    if not isinstance(program["id"], str) or not program["id"]:
        raise ValueError("%s.id 必须是非空字符串" % name)
    if not isinstance(program["doc"], str):
        raise ValueError("%s.doc 必须是字符串" % name)
    _as_int(program["size"], "%s.size" % name, 0)
    return {"id": program["id"], "doc": program["doc"], "size": int(program["size"])}


def _program_probs(programs):
    """簇内取样概率：长度归一化后取 softmax(-x)。

    官方实现的归一化分母是 `max(lengths) + 1e-6`（不是 `max - min`），
    且 softmax 的温度恒为 1.0——长度越短概率越高，但不会塌成 argmin。
    """
    lengths = [p["size"] for p in programs]
    lo, hi = min(lengths), max(lengths)
    denom = hi + 1e-6
    normalized = [(x - lo) / denom for x in lengths]
    return _softmax([-x for x in normalized], 1.0)


def solve(events, num_islands=10, functions_per_prompt=2,
          cluster_sampling_temperature_init=0.1,
          cluster_sampling_temperature_period=30000, function_name="f"):
    n_islands = _as_int(num_islands, "num_islands", 1)
    per_prompt = _as_int(functions_per_prompt, "functions_per_prompt", 1)
    temp_init = _as_real(cluster_sampling_temperature_init,
                         "cluster_sampling_temperature_init")
    if temp_init <= 0:
        raise ValueError("cluster_sampling_temperature_init 必须为正")
    period = _as_int(cluster_sampling_temperature_period,
                     "cluster_sampling_temperature_period", 1)
    if not isinstance(function_name, str) or not function_name:
        raise ValueError("function_name 必须是非空字符串")
    if not isinstance(events, (list, tuple)):
        raise ValueError("events 必须是列表或元组")

    islands = [{"order": [], "index": {}, "num_programs": 0}
               for _ in range(n_islands)]
    best_score = [-math.inf] * n_islands
    best_program = [None] * n_islands
    best_tests = [None] * n_islands
    registrations, prompts, resets = [], [], []

    def register(program, island_id, scores, via):
        island = islands[island_id]
        signature = _signature(scores)
        score = _reduce(scores)
        pos = island["index"].get(signature)
        new_cluster = pos is None
        if new_cluster:
            pos = len(island["order"])
            island["order"].append({"signature": list(signature), "score": score,
                                    "programs": [dict(program)]})
            island["index"][signature] = pos
        else:
            island["order"][pos]["programs"].append(dict(program))
        island["num_programs"] += 1
        improved = score > best_score[island_id]
        if improved:
            best_score[island_id] = score
            best_program[island_id] = dict(program)
            best_tests[island_id] = dict(scores)
        registrations.append({"program": program["id"], "island": island_id,
                              "cluster": pos, "new_cluster": new_cluster,
                              "score": score, "improved": improved,
                              "via": via})

    def do_reset(founders):
        count = n_islands // 2
        if len(founders) != count:
            raise ValueError("重置 %d 个岛，需要 %d 个创始人" % (count, count))
        order = sorted(range(n_islands), key=lambda i: (best_score[i], i))
        reset_ids, keep_ids = order[:count], order[count:]
        picks = []
        for raw in founders:
            founder = _as_int(raw, "founders 项", 0)
            if founder >= n_islands or founder not in keep_ids:
                raise ValueError("创始人岛必须是保留岛")
            if best_program[founder] is None:
                raise ValueError("创始人岛还没有任何程序")
            picks.append(founder)
        for island_id, founder in zip(reset_ids, picks):
            islands[island_id] = {"order": [], "index": {}, "num_programs": 0}
            best_score[island_id] = -math.inf
            best_program[island_id] = None
            best_tests[island_id] = None
            register(best_program[founder], island_id, best_tests[founder], "reset")
        resets.append({"reset": list(reset_ids), "keep": list(keep_ids),
                       "founders": {i: f for i, f in zip(reset_ids, picks)}})

    for raw in events:
        if not isinstance(raw, dict):
            raise ValueError("events 每项必须是字典")
        kind = raw.get("kind")
        if kind not in _KINDS:
            raise ValueError("未知的 kind: %r" % (kind,))
        if kind == "register":
            program = _as_program(raw.get("program"), "program")
            scores = raw.get("scores")
            if not isinstance(scores, dict) or not scores:
                raise ValueError("scores 必须是非空字典")
            clean = {}
            for key, value in scores.items():
                if not isinstance(key, str):
                    raise ValueError("scores 的键必须是字符串")
                clean[key] = _as_real(value, "scores[%r]" % key)
            target = raw.get("island")
            if target is None:
                for island_id in range(n_islands):
                    register(program, island_id, clean, "broadcast")
            else:
                island_id = _as_int(target, "island", 0)
                if island_id >= n_islands:
                    raise ValueError("island 越界")
                register(program, island_id, clean, "event")
        elif kind == "prompt":
            island_id = _as_int(raw.get("island"), "island", 0)
            if island_id >= n_islands:
                raise ValueError("island 越界")
            clusters = raw.get("clusters")
            picked = raw.get("programs")
            if not isinstance(clusters, (list, tuple)) or not isinstance(picked, (list, tuple)):
                raise ValueError("clusters / programs 必须是序列")
            island = islands[island_id]
            total = len(island["order"])
            if total == 0:
                raise ValueError("空岛没有可用的簇")
            take = min(total, per_prompt)
            if len(clusters) != take or len(picked) != take:
                raise ValueError("取样簇数必须是 min(簇数, functions_per_prompt)=%d" % take)
            chosen = []
            for c_idx, p_idx in zip(clusters, picked):
                c_idx = _as_int(c_idx, "clusters 项", 0)
                if c_idx >= total:
                    raise ValueError("簇下标越界")
                progs = island["order"][c_idx]["programs"]
                p_idx = _as_int(p_idx, "programs 项", 0)
                if p_idx >= len(progs):
                    raise ValueError("程序下标越界")
                chosen.append((c_idx, p_idx))
            temperature = temp_init * (1 - (island["num_programs"] % period) / period)
            scores = [c["score"] for c in island["order"]]
            probabilities = _softmax(scores, temperature)
            order = sorted(range(take),
                           key=lambda i: (island["order"][chosen[i][0]]["score"], i))
            versions = []
            for slot, idx in enumerate(order):
                c_idx, p_idx = chosen[idx]
                source = island["order"][c_idx]["programs"][p_idx]
                if slot == 0:
                    doc = source["doc"]
                else:
                    doc = "Improved version of `%s_v%d`." % (function_name, slot - 1)
                versions.append({"name": "%s_v%d" % (function_name, slot),
                                 "docstring": doc, "source": source["id"]})
            prompts.append({
                "island": island_id,
                "temperature": temperature,
                "num_programs": island["num_programs"],
                "cluster_scores": scores,
                "probabilities": probabilities,
                "chosen_clusters": [c for c, _ in chosen],
                "chosen_programs": [p for _, p in chosen],
                "program_probs": [_program_probs(island["order"][c]["programs"])
                                  for c, _ in chosen],
                "sorted_clusters": [chosen[i][0] for i in order],
                "versions": versions,
                "version_generated": take + 1,
                "header": {"name": "%s_v%d" % (function_name, take),
                           "docstring": "Improved version of `%s_v%d`."
                                        % (function_name, take - 1)},
            })
        else:
            founders = raw.get("founders")
            if not isinstance(founders, (list, tuple)):
                raise ValueError("founders 必须是序列")
            do_reset(founders)

    return {
        "clusters": [[{"signature": c["signature"], "score": c["score"],
                       "programs": [p["id"] for p in c["programs"]]}
                      for c in island["order"]] for island in islands],
        "cluster_count": [len(island["order"]) for island in islands],
        "num_programs": [island["num_programs"] for island in islands],
        "best_score": list(best_score),
        "best_program": list(best_program),
        "registrations": registrations,
        "prompts": prompts,
        "resets": resets,
    }
