"""Auto-CoT 的聚类选代表拼 demo（单阶段、确定性核心）。

官方 amazon-science/auto-cot 的 run_demo.py 确定性核心：
（SentenceTransformer 编码与 KMeans 拟合不入判题域——retrieval-kmeans-lloyd
已占 k-means 算子；聚类分配 cluster_assignment 与到各簇质心的欧氏距离
矩阵 dist 作显式入参，对应官方 clustering_model.labels_ 与
clustering_model.transform(corpus_embeddings)。）

1. 分簇：第 i 簇收集所有 assignment==i 的样本，记录到质心 i 的距离
   与全局下标（官方 clustered_dists/clustered_idx，按下标升序 append）；
2. 簇内排序：按距离**稳定升序**（官方 sorted 默认 stable，距离平局保持
   全局下标序）；
3. 逐候选过滤（官方五重条件，任一失败换下一候选）：
   a. 问题词数 len(question.strip().split()) <= 60；
   b. rationale 行数 len(r.replace("\\n\\n", "\\n").split("\\n")) <= max_ra_len
      （r = rationale.strip()；last_letters 类任务由调用方传 max_ra_len=7）；
   c. rationale 以 "." 结尾（官方 c_rationale[-1]=="."；空串视为不通过）；
   d. pred_ans strip 后非空；
   e. arithmetic=True 时算术答案回验：pred_ans 必须出现在 rationale 的
      倒数第二个句子（r.split(".")[-2]，前置条件 c 保证 split 至少 2 段）
      或最后 10 个词（r.split()[-10:]）中，否则跳过；
4. 通过者收为该簇 demo 并 break（**每簇至多一个**；官方 sampling=center
   路径，random.shuffle 变体不入判题域）；
5. demo 组装：rationale 清洗 = replace("\\n\\n","\\n") → replace("\\n"," ")
   → strip → " ".join(split())（空白折叠成单空格）；question/pred_ans
   原样（官方的 "\\nA:" 后缀与 "A: Let's think..." 替换属 log 解析层）；
   gold_anses 给出时 strip 透传，None 时为 null；
6. 簇按编号 0..k-1 依次处理，某簇全被过滤则该簇无 demo（空簇同样）。

边界：log 文件解析（Q:/A:/pred_mode/GT : 行协议）、PCA 可视化、
random.shuffle 采样均不在判题域；样本从零开始编号。
"""

_EMPTY = ""


def _validate(name, value, n=None):
    if not isinstance(value, list):
        raise ValueError(f"{name} 必须是 list")
    if not value:
        raise ValueError(f"{name} 不能为空")
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{name}[{i}] 必须是字符串")
    if n is not None and len(value) != n:
        raise ValueError(f"{name} 长度必须为 {n}")


def solve(questions, rationales, pred_anses, cluster_assignment, dist,
          max_ra_len=5, arithmetic=True, gold_anses=None):
    n = len(questions)
    _validate("questions", questions)
    _validate("rationales", rationales, n)
    _validate("pred_anses", pred_anses, n)
    if not isinstance(cluster_assignment, list) or len(cluster_assignment) != n:
        raise ValueError("cluster_assignment 必须是与样本等长的 list")
    for i, a in enumerate(cluster_assignment):
        if isinstance(a, bool) or not isinstance(a, int):
            raise ValueError(f"cluster_assignment[{i}] 必须是整数")
    if not isinstance(dist, list) or len(dist) != n:
        raise ValueError("dist 必须是 (n, k) 形状的 list")
    k = 0
    for i, row in enumerate(dist):
        if not isinstance(row, list) or not row:
            raise ValueError(f"dist[{i}] 必须是非空 list")
        if i == 0:
            k = len(row)
        elif len(row) != k:
            raise ValueError(f"dist[{i}] 宽度必须为 {k}")
        for j, v in enumerate(row):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"dist[{i}][{j}] 必须是数值")
    if k == 0:
        raise ValueError("dist 必须至少含一个簇")
    for i, a in enumerate(cluster_assignment):
        if not 0 <= a < k:
            raise ValueError(f"cluster_assignment[{i}]={a} 越界（簇数 {k}）")
    if isinstance(max_ra_len, bool) or not isinstance(max_ra_len, int) \
            or max_ra_len < 1:
        raise ValueError("max_ra_len 必须是 >= 1 的整数")
    if not isinstance(arithmetic, bool):
        raise ValueError("arithmetic 必须是 bool")
    if gold_anses is not None:
        _validate("gold_anses", gold_anses, n)

    # 1. 分簇（按下标升序 append，与官方循环一致）
    clustered = [[] for _ in range(k)]  # [(dist, global_idx), ...]
    for sid, a in enumerate(cluster_assignment):
        clustered[a].append((dist[sid][a], sid))

    demos = []
    selected = []
    cluster_order = []
    # 2-6. 每簇按编号序处理，簇内距离稳定升序，逐候选过滤取第一个通过者
    for ci in range(k):
        candidates = sorted(clustered[ci], key=lambda t: (t[0], t[1]))
        for _, sid in candidates:
            question = questions[sid]
            r = rationales[sid].strip()
            p = pred_anses[sid].strip()
            if len(question.strip().split()) > 60:
                continue
            if len(r.replace("\n\n", "\n").split("\n")) > max_ra_len:
                continue
            if not r.endswith("."):
                continue
            if p == _EMPTY:
                continue
            if arithmetic \
                    and not (p in r.split(".")[-2] or p in r.split()[-10:]):
                continue
            rationale = " ".join(r.replace("\n\n", "\n")
                                 .replace("\n", " ").strip().split())
            demos.append({
                "question": question,
                "rationale": rationale,
                "pred_ans": p,
                "gold_ans": None if gold_anses is None
                else gold_anses[sid].strip(),
            })
            selected.append(sid)
            cluster_order.append(ci)
            break
    return {"demos": demos, "selected": selected,
            "cluster_order": cluster_order, "num_clusters": k}
