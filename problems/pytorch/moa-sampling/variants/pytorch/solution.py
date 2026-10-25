"""参考答案：MoA（Mixture-of-Adaptive-samplers）多路过采样 + 加权混合后单次采样。

机制（与单采样器题的区别：分布源本身被混合，不是只调温度）：
1. 每个专家 e 先按自己的温度出分布：temp<=0 时退化为 one-hot(argmax)（llama.cpp
   `llama_sampler_temp_impl` 的口径），否则 softmax(logits_e / temp_e)；再过自己的自适应
   截断器 top_p_e（降序累积和首次 >= p 的前缀，至少 1 个；并列按索引升序），归一化得 q_e。
2. 过采样：专家 e 用 u_experts[e, j]（j=1..m）在 q_e 的**原顺序** CDF 上各逆变换抽 1 个 token，
   共 m 次（有放回）。所有专家的 m·E 次抽样的并集 = 联合候选集 U（去重、按索引升序）。
3. 每个专家在 U 上再次归一化：r_e(t) = q_e(t) / sum_{t' in U} q_e(t')；若专家在 U 上总质量为 0
   （例如该专家 m=0），它对混合不贡献。
4. 加权混合：p_mix(t) = sum_e (w_e / sum_w) · r_e(t)，再在 U 上归一化成 filtered（U 外为 0）。
5. 单次采样：用 u 在 filtered 原顺序 CDF 上逆变换得 chosen。

返回 (filtered, chosen, in_union)；in_union 为 bool 张量 (V,)，标记联合候选集。
"""

import torch


def _inverse_cdf_row(dist, u):
    cdf = torch.cumsum(dist, dim=-1)
    ge = cdf >= float(u)
    if not bool(ge.any()):
        return dist.shape[-1] - 1
    return int(torch.nonzero(ge)[0, 0])


def _nucleus_row(p, top_p):
    """降序累积和首次 >= top_p 的前缀（至少 1 个），返回归一化后的整行分布。"""
    v = p.shape[-1]
    order = torch.argsort(p, dim=-1, descending=True, stable=True)
    sp = p[order]
    cum = torch.cumsum(sp, dim=-1)
    keep_rank = (cum - sp) < float(top_p)
    if not bool(keep_rank.any()):
        keep_rank[0] = True
    kept = sp * keep_rank.to(p.dtype)
    kept = kept / kept.sum()
    out = torch.zeros_like(p)
    out[order] = kept
    return out


def solve(logits, weights, temps, top_ps, m, u_experts, u):
    e_n, v = logits.shape
    m = int(m)
    q = torch.zeros(e_n, v)
    for e in range(e_n):
        t = float(temps[e])
        if t <= 0.0:
            p = torch.zeros(v)
            p[int(torch.argmax(logits[e]))] = 1.0
        else:
            p = torch.softmax(logits[e] / t, dim=-1)
        q[e] = _nucleus_row(p, float(top_ps[e]))

    union = set()
    for e in range(e_n):
        for j in range(m):
            union.add(_inverse_cdf_row(q[e], u_experts[e][j]))
    u_idx = sorted(union)

    w = torch.tensor([float(x) for x in weights], dtype=torch.float32)
    w = w / w.sum()

    mix = torch.zeros(v)
    for e in range(e_n):
        denom = q[e, u_idx].sum() if u_idx else torch.tensor(0.0)
        if float(denom) <= 0.0:
            continue
        for t_id in u_idx:
            mix[t_id] += float(w[e]) * float(q[e, t_id]) / float(denom)

    total = mix[u_idx].sum() if u_idx else torch.tensor(0.0)
    filtered = torch.zeros(v)
    if float(total) > 0.0:
        filtered[u_idx] = mix[u_idx] / total

    in_union = torch.zeros(v, dtype=torch.bool)
    in_union[u_idx] = True
    return filtered, torch.tensor(_inverse_cdf_row(filtered, u), dtype=torch.long), in_union
