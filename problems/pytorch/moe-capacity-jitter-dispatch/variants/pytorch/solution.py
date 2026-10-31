"""参考答案：先算候选和优先级键，再按全局键序贪心占位，最后在存活槽位上归一化。

四个容易写错的点：

1. **jitter 只改处理顺序，不改候选集合**。top-k 只看 `probs`；如果把 `probs + jitter`
   重新排序取前 k，候选集合本身就变了，等于另一道题（那是 noisy top-k 门控的口径）。
2. **处理顺序是全局降键**，同一 token 的低秩槽位可能先被处理。于是「本 token 已占用哪个
   专家」必须随占位动态更新，否则同一个 token 的两个槽位会挤在同一个专家上；反过来，
   高秩槽位先占位成功后，低秩槽位就只能往前溢出或丢弃。
3. **`assign == -1` 不能直接进 `gather`**：PyTorch 的负索引会回绕到最后一个专家，
   被丢弃的槽位会悄悄拿到一个真实概率，归一化分母就错了。要么先 `clamp(min=0)` 再
   `where` 置零，要么用掩码乘。
4. **归一化只覆盖存活的槽位**。丢弃位置贡献 0，权重和为 1 是针对「留下的那几个专家」说的，
   不是针对原来的 top-k。全行丢弃时分母为 0，必须显式给 0 而不是让除零产生 NaN。

贪心本身有顺序依赖（后一个槽位能不能占位取决于前面占了什么），所以这里就写成循环；
真正能向量化的部分只有候选选取和键排序。
"""

import torch


def _check(probs, k, capacity, jitter):
    for name, tensor in (("probs", probs), ("jitter", jitter)):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor")
        if tensor.device.type != "cpu":
            raise ValueError(f"{name} must be on CPU")
        if tensor.dtype != torch.float64:
            raise ValueError(f"{name} must be float64")
    if probs.dim() != 2 or min(probs.shape) < 1:
        raise ValueError("probs must be a non-empty 2-D [T, E] tensor")
    if jitter.shape != probs.shape:
        raise ValueError("jitter must have the same shape as probs")
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError("k must be a python int")
    if isinstance(capacity, bool) or not isinstance(capacity, int):
        raise ValueError("capacity must be a python int")
    if not 1 <= k <= probs.shape[1]:
        raise ValueError("k must be in [1, E]")
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    if not bool(torch.isfinite(probs).all()) or bool((probs < 0.0).any()):
        raise ValueError("probs must be finite and non-negative")
    if not bool(torch.isfinite(jitter).all()):
        raise ValueError("jitter must be finite")


def solve(probs, k, capacity, jitter):
    _check(probs, k, capacity, jitter)
    num_tokens, num_experts = probs.shape

    # 候选：稳定降序排序的前 k 个，并列时索引小的在前
    sorted_probs, sorted_ids = torch.sort(probs, dim=-1, descending=True, stable=True)
    cand = sorted_ids[:, :k]
    key = sorted_probs[:, :k] + jitter.gather(1, cand)
    order = torch.argsort(-key.reshape(-1), stable=True)

    remaining = [capacity] * num_experts
    used = [set() for _ in range(num_tokens)]
    cand_rows = cand.tolist()
    assign = torch.full((num_tokens, k), -1, dtype=torch.int64)
    for slot in order.tolist():
        token, rank = divmod(int(slot), k)
        for expert in cand_rows[token][rank:]:        # 先自己的专家，再沿秩向前溢出
            expert = int(expert)
            if remaining[expert] <= 0 or expert in used[token]:
                continue
            assign[token, rank] = expert
            remaining[expert] -= 1
            used[token].add(expert)
            break

    # -1 先进 clamp 再被 where 置零：负索引会回绕成最后一个专家的概率
    kept = assign >= 0
    kept_probs = torch.where(kept, probs.gather(1, assign.clamp(min=0)), 0.0)
    denom = kept_probs.sum(dim=1, keepdim=True)
    weights = torch.where(denom > 0.0, kept_probs / denom, 0.0)

    counts = torch.bincount(assign[kept], minlength=num_experts)[:num_experts]
    dropped = int(num_tokens * k) - int(kept.sum())
    return assign, weights, counts, dropped, order
