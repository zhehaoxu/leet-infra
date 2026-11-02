def solve(x, plan, n_select_sample=5):
    """ToT + BFS 逐步账（Tree of Thoughts, 2305.10601 官方 princeton-nlp/
    tree-of-thought-llm 的 src/tot/methods/bfs.py::solve + game24 价值头）。

    LLM 生成与评估全部作显式入参：
    - plan[r]["method"] = "propose" | "sample"
      * propose: gen[i] 为父节点 i 的一条模型输出，按 "\\n" 切成 proposals，
        子候选 = y + p + "\\n"（官方含尾部换行）
      * sample: gen[i] 为父节点 i 的 n_generate_sample 条样本，
        子候选 = y + s（无分隔符）
    - evals[i][j] 为父节点 i 第 j 个子候选的评估文本列表（game24 价值头）
    - n_select_sample 即论文里的 b

    每步流程：展开子候选（父节点优先序）→ 打分（重复候选直接 0；
    值 = 评估文本末行查 {impossible:0.001, likely:1, sure:20} 加权和，
    不除以样本数；候选恰 4 行且无 answer 直接 0）→ greedy 稳定降序取前 b
    （平分保原下标序）→ 本轮选择即下一轮父节点。

    返回 {"final": 末轮选择, "steps": 逐步账}，每步账含
    step / x / ys / new_ys / values / select_new_ys。

    域约束：n_select_sample 为正整数；plan 非空且每步含 method/gen/evals；
    method 非法、槽位缺 gen/evals、evals[i] 与子候选数不匹配 → ValueError。
    （官方对未知 method_generate 是 NameError；vote 评估与 sample 选择
    因随机性不入题。）
    """
    raise NotImplementedError
