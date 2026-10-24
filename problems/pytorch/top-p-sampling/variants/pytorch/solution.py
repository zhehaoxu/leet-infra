"""参考答案：Top-p / Nucleus 采样（把随机性作为显式输入以保证判题确定性）。

流程：
1. probs = softmax(logits)。
2. 按概率降序排序，做累积和 cumsum。
3. 核(nucleus)：取「累积和首次达到 p」的最小前缀（至少保留概率最大的 1 个 token）；
   判据 (cumsum - sorted_prob) < p 等价于保留到跨过 p 的那个元素为止。
4. 核内概率重新归一化，映射回原始顺序得到 filtered 分布（核外为 0）。
5. 用输入的均匀随机数 u∈[0,1) 对核内分布做逆变换采样（CDF 上首个 >= u 的位置），
   返回被选中 token 的原始索引。

返回 (filtered, chosen_index)。
"""

import torch


def solve(logits, p, u):
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, idx = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    # 保留到累积和首次 >= p 的元素为止（其左侧 cumsum - 当前 < p）
    keep = (cumsum - sorted_probs) < p
    nucleus = sorted_probs * keep.float()
    nucleus = nucleus / nucleus.sum(dim=-1, keepdim=True)

    # 映射回原始顺序
    filtered = torch.zeros_like(probs)
    filtered.scatter_(-1, idx, nucleus)

    # 把 u 补成可在最后一维广播的形状：标量→(1,)，(batch,)→(batch,1)
    if u.dim() < probs.dim():
        u = u.view(u.shape + (1,) * (probs.dim() - u.dim()))

    # 逆变换采样：在核内(降序)CDF 上找首个 >= u 的位置
    cdf = torch.cumsum(nucleus, dim=-1)
    hit = cdf >= u
    pos = hit.to(torch.long).argmax(dim=-1)
    # 浮点误差兜底：若没有命中（u 略大于末位 CDF），取核内最后一个保留元素
    none_hit = hit.sum(dim=-1) == 0
    if bool(none_hit.any()):
        last = keep.float().sum(dim=-1).long() - 1
        pos = torch.where(none_hit, last, pos)
    chosen = idx.gather(-1, pos.unsqueeze(-1)).squeeze(-1)
    return filtered, chosen
