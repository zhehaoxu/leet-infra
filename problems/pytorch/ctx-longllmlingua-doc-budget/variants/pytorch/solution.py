import re


def solve(doc_token_ids, doc_ppl, token_loss, target_token, context_budget="+100",
          force_context_ids=None, reorder_context="original",
          dynamic_context_compression_ratio=0.0, condition_in_question="none",
          iterative_size=200):
    """LongLLMLingua 的文档预算、动态压缩率与窗口阈值计划（2310.06839，CPU 参考口径）。"""
    def _fail(msg):
        raise ValueError(msg)

    # ---- 类型与取值校验 ----
    if not isinstance(doc_token_ids, (list, tuple)) or not doc_token_ids:
        _fail("doc_token_ids 必须是非空 list")
    docs = []
    for i, doc in enumerate(doc_token_ids):
        if not isinstance(doc, (list, tuple)) or not doc:
            _fail(f"doc_token_ids[{i}] 必须是非空 list")
        row = []
        for j, t in enumerate(doc):
            if isinstance(t, bool) or not isinstance(t, int):
                _fail(f"doc_token_ids[{i}][{j}] 必须是 int")
            row.append(t)
        docs.append(row)
    n_docs = len(docs)

    if not isinstance(doc_ppl, (list, tuple)) or len(doc_ppl) != n_docs:
        _fail("doc_ppl 必须与 doc_token_ids 等长")
    ppl = []
    for i, p in enumerate(doc_ppl):
        if isinstance(p, bool) or not isinstance(p, (int, float)) or p != p or p in (float("inf"), float("-inf")):
            _fail(f"doc_ppl[{i}] 必须是有限实数")
        ppl.append(float(p))

    total = sum(len(d) for d in docs)
    if not isinstance(token_loss, (list, tuple)) or len(token_loss) != total:
        _fail("token_loss 长度必须等于所有文档 token 数之和")
    loss = []
    for j, v in enumerate(token_loss):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
            _fail(f"token_loss[{j}] 必须是有限实数")
        loss.append(float(v))

    if isinstance(target_token, bool) or not isinstance(target_token, (int, float)):
        _fail("target_token 必须是实数")
    target_token = float(target_token)

    if not isinstance(context_budget, str) or not re.match(r"^[+\-*/]\d+(\.\d+)?$", context_budget):
        _fail("context_budget 必须是形如 '+100' 的四则运算串")
    if reorder_context not in ("original", "two_stage"):
        _fail("reorder_context 只能是 'original' 或 'two_stage'")
    if isinstance(dynamic_context_compression_ratio, bool) or not isinstance(dynamic_context_compression_ratio, (int, float)):
        _fail("dynamic_context_compression_ratio 必须是实数")
    lam = abs(float(dynamic_context_compression_ratio))
    if condition_in_question not in ("none", "question"):
        _fail("condition_in_question 只能是 'none' 或 'question'")
    if isinstance(iterative_size, bool) or not isinstance(iterative_size, int) or iterative_size < 1:
        _fail("iterative_size 必须是 >= 1 的 int")

    force = []
    if force_context_ids is not None:
        if not isinstance(force_context_ids, (list, tuple)):
            _fail("force_context_ids 必须是 list 或 None")
        for i, f in enumerate(force_context_ids):
            if isinstance(f, bool) or not isinstance(f, int):
                _fail(f"force_context_ids[{i}] 必须是 int")
            if not 0 <= f < n_docs:
                _fail(f"force_context_ids[{i}] 越界")
            force.append(f)

    # ---- ① 文档排序：question 感知 PPL 显式给出，按方向稳定排序 ----
    sort_direct = -1 if condition_in_question == "none" else 1
    rank_order = [i for i, _ in sorted(enumerate(ppl), key=lambda x: sort_direct * x[1])]

    # ---- ② 预算账：target_token<0 先归 100，再套 context_budget 四则 ----
    if target_token < 0:
        target_token = 100.0
    op, num = context_budget[0], float(context_budget[1:])
    if op == "+":
        target_token = target_token + num
    elif op == "-":
        target_token = target_token - num
    elif op == "*":
        target_token = target_token * num
    else:
        target_token = target_token / num

    used = list(force)
    budget = target_token  # 上游 compress_prompt 传给窗口阶段的是求值后的原值（扣减只在局部）
    for idx in rank_order:
        budget -= len(docs[idx])
        if idx not in used:
            used.append(idx)
        if budget < 0:
            break
    original_used = used

    if reorder_context == "original":
        used = sorted(used)
    else:
        evens = [x for i, x in enumerate(used) if i % 2 == 0]
        odds = [x for i, x in enumerate(used) if i % 2 == 1]
        used = evens + odds[::-1]

    # ---- ③ 动态压缩率：i 走 -(N-1)..(N-1) 步长 2，再整体反转，按文档下标回填 ----
    n_used = len(original_used)
    if lam > 0:
        vals = [i * (lam / (n_used - 1)) if n_used > 1 else 0.0
                for i in range(-(n_used - 1), n_used, 2)][::-1]
        ratio_map = dict(zip(original_used, vals))
        dynamic_ratios = [ratio_map[i] for i in used]
    else:
        dynamic_ratios = [0.0] * len(used)

    # ---- ⑤ 窗口计划：预算用原始长度、窗口用 len+2（上游口径），tau 均摊 ----
    context_length = [len(docs[i]) + 2 for i in used]
    tau = target_token / (sum(context_length) + 1)
    def _clamp01(x):
        return max(min(1.0, x), 0.0)

    doc_ratios = [_clamp01(tau + dr) for dr in dynamic_ratios]

    window_plan = []
    idx_doc, last, cur = 0, 1, []
    while idx_doc < len(context_length):
        seg_len, seg_ratio = context_length[idx_doc], doc_ratios[idx_doc]
        if last + seg_len >= iterative_size:
            cur.append([iterative_size - last, seg_ratio])
            window_plan.append(cur)
            last = last + seg_len - iterative_size
            if last > iterative_size:
                k = last // iterative_size
                window_plan.extend([[[iterative_size, seg_ratio]] for _ in range(k)])
                last -= k * iterative_size
            cur = [[last, seg_ratio]] if last else []
        else:
            last += seg_len
            cur.append([seg_len, seg_ratio])
        idx_doc += 1
    if cur:
        window_plan.append(cur)

    # ---- ④ 阈值：过滤 10000 哨兵后取降序第 int(len*ratio)-1 位；ratio==1 全留 ----
    filtered = [v for v in loss if v != 10000.0]
    thresholds = []
    keep_masks = []
    block = 0
    for window in window_plan:
        for delta_end, ratio in window:
            if ratio == 1.0:
                threshold = float("-inf")
            else:
                pos = max(0, min(len(filtered) - 1, int(len(filtered) * ratio) - 1))
                threshold = sorted(filtered, reverse=True)[pos]
            mask = [True] * total
            start = block * iterative_size
            for j in range(start, min(start + delta_end, total)):
                if j > 0:
                    mask[j] = loss[j] > threshold
            thresholds.append(threshold)
            keep_masks.append(mask)
            block += 1

    return {
        "rank_order": rank_order,
        "selected": used,
        "dynamic_ratios": dynamic_ratios,
        "tau": tau,
        "window_plan": window_plan,
        "thresholds": thresholds,
        "keep_masks": keep_masks,
    }
