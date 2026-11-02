# Tree of Thoughts (Yao et al., 2305.10601) 官方 princeton-nlp/tree-of-thought-llm
# src/tot/methods/bfs.py 的 solve 主循环 + src/tot/tasks/game24.py 的
# value_outputs_unwrap 标量化。LLM 生成与评估全部作显式入参：
#   - plan[r]["method"] = "propose" | "sample"（官方 method_generate）
#   - gen: 逐父节点槽位的生成结果（propose 为单条输出按 \n 切分出 proposals；
#          sample 为 n_generate_sample 条样本）
#   - evals: 逐父节点逐子候选的 n_evaluate_sample 条评估文本（game24 价值头）
# 官方不建模（记 notes）：task.value_cache 跨步缓存、method_evaluate='vote'、
# method_select='sample'（np.random.choice 全局 RNG）、未知的 method_generate
# 触发官方 NameError（本题改 ValueError）。


def _unwrap_value(y, eval_outputs):
    """game24 value_outputs_unwrap：4 行且无 answer 直接 0；
    否则按各评估文本最后一行查 ad-hoc 权重表求和（不除以样本数）。"""
    if len(y.strip().split("\n")) == 4 and "answer" not in y.lower():
        return 0
    value_names = [_.split("\n")[-1] for _ in eval_outputs]
    value_map = {"impossible": 0.001, "likely": 1, "sure": 20}  # TODO: ad hoc（官方原注）
    value = sum(v * value_names.count(name) for name, v in value_map.items())
    return value


def solve(x, plan, n_select_sample=5):
    """ToT + BFS 逐步账（官方 solve，评估走 value 分支、选择走 greedy 分支）。

    返回 {"final": 末轮 select_new_ys, "steps": 逐步 infos 账}，每步账为
    {"step", "x", "ys"(本轮父节点), "new_ys", "values", "select_new_ys"}。
    """
    if not isinstance(x, str):
        raise ValueError("x must be a str")
    if isinstance(n_select_sample, bool) or not isinstance(n_select_sample, int) \
            or n_select_sample < 1:
        raise ValueError("n_select_sample must be a positive int (b in the paper)")
    if not isinstance(plan, (list, tuple)) or len(plan) == 0:
        raise ValueError("plan must be a non-empty list of step dicts")

    ys = [""]  # 官方初始候选
    infos = []
    for step, rnd in enumerate(plan):
        if not isinstance(rnd, dict) or not {"method", "gen", "evals"} <= set(rnd):
            raise ValueError("plan step entries need keys method/gen/evals")
        method = rnd["method"]
        gen = rnd["gen"]
        evals = rnd["evals"]
        if method not in ("propose", "sample"):
            # 官方无 else 分支，未知方法在下一行触发 NameError；本题显式拒收
            raise ValueError("method must be 'propose' or 'sample'")

        new_ys = []       # 展平后的候选（父节点优先序，itertools.chain 语义）
        child_evals = []  # 与 new_ys 对齐的评估文本列表
        for i, y in enumerate(ys):
            if i >= len(gen) or i >= len(evals):
                raise ValueError(f"step {step} needs gen/evals for {len(ys)} parents")
            if method == "propose":
                if not isinstance(gen[i], str):
                    raise ValueError("propose gen[i] must be a single output str")
                proposals = gen[i].split("\n")  # 官方 get_proposals 的切分
                if len(evals[i]) != len(proposals):
                    raise ValueError("propose evals[i] must match proposal count")
                for j, p in enumerate(proposals):
                    new_ys.append(y + p + "\n")  # 官方拼接含尾部换行
                    child_evals.append(evals[i][j])
            else:  # sample
                if not isinstance(gen[i], (list, tuple)):
                    raise ValueError("sample gen[i] must be a list of sample strs")
                if len(evals[i]) != len(gen[i]):
                    raise ValueError("sample evals[i] must match sample count")
                for j, s in enumerate(gen[i]):
                    new_ys.append(y + s)  # 官方拼接无分隔符
                    child_evals.append(evals[i][j])

        # 评估：官方 get_values——同一候选重复出现时直接记 0（不是缓存值）
        values = []
        local_value_cache = {}
        for c, y in enumerate(new_ys):
            if y in local_value_cache:
                value = 0
            else:
                value = _unwrap_value(y, child_evals[c])
                local_value_cache[y] = value
            values.append(value)

        # 选择：官方 greedy——稳定排序降序，平分保持原下标序（小 id 优先）
        ids = list(range(len(new_ys)))
        select_ids = sorted(ids, key=lambda t: values[t], reverse=True)[:n_select_sample]
        select_new_ys = [new_ys[t] for t in select_ids]

        infos.append({"step": step, "x": x, "ys": list(ys),
                      "new_ys": list(new_ys), "values": values,
                      "select_new_ys": list(select_new_ys)})
        ys = select_new_ys  # 本轮选择即下一轮父节点

    return {"final": ys, "steps": infos}
