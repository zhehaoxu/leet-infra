"""参考答案：Contrastive Search（CNTRM）逐步选择（确定性 argmax，无内部随机数）。

对齐论文公式（arXiv 2202.06417, Wu et al. 2022）
M(h_t; λ, t) = max_{t'<=t} { (1−λ)·log p_t − λ·d(h_t, h_{t'}) }，d = (1+cos)/2 ∈ [0,1]：
由于 (1−λ)·log p_t 与 t' 无关，max 可提到 deg_t = max_{t'} d_t t' 上，于是
score_j = (1−λ)·log p_j − λ·deg_j。

流程：
1. 候选 = 按 softmax 概率降序的 top-k（并列按索引升序）；`sim` 的第 j 行即该候选的隐态
   与历史（已选 token）隐态的余弦相似度，故行序必须与降序候选对齐。
2. deg_j = max_i (sim[j,i] + 1)/2；无历史（T=0）时 deg_j = 0。夹到 [0,1]。
3. score_j = (1−λ)·log_softmax(logits)[cand_j] − λ·deg_j（log 用全词表 softmax，不做核内归一）。
4. 选 argmax(score)，并列取候选顺序里更靠前的（即 logprob 更大者）——这就是「逐步打分选择」，
   被选中的 token 的隐态进入下一步的历史，退化惩罚随重复变严重而抬升。

返回 (cand_idx, deg, scores, chosen)：cand_idx/int64 (k,)、deg/scores/float32 (k,)、chosen 标量。
"""

import torch


def solve(logits, sim, k, lam):
    probs = torch.softmax(logits, dim=-1)
    v = probs.shape[-1]
    k = min(int(k), v)

    order = torch.argsort(probs, dim=-1, descending=True, stable=True)
    cand_idx = order[:k].to(torch.long)

    log_probs = torch.log_softmax(logits, dim=-1)
    logp = log_probs[cand_idx]

    lam = float(lam)
    t = int(sim.shape[-1]) if sim.dim() > 1 else 0
    if t > 0:
        d01 = (sim[:k].reshape(k, t) + 1.0) / 2.0
        deg = d01.max(dim=-1).values.clamp(0.0, 1.0)
    else:
        deg = torch.zeros(k, dtype=probs.dtype)

    scores = (1.0 - lam) * logp - lam * deg
    # argmax，并列取更靠前的候选（与 torch.argmax(CPU) 的首次命中一致）
    pos = 0
    best = float(scores[0])
    for j in range(1, k):
        if float(scores[j]) > best:
            best, pos = float(scores[j]), j

    return cand_idx, deg.to(probs.dtype), scores, cand_idx[pos]
