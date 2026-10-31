"""参考解：照 an-yongqi/rest-kv 的 restkv/restkv_utils.py::RestKVCluster.update_kv L584-699。

adaptive 池化（L662-674）与 use_pyramid 的逐层预算（L591-602、L683-684）不进本题：
前者硬编码 batch 0 且核宽由注意力排序差算出，后者与 kv-pyramid-layer-budget 重复。
上游 L608 造掩码时没传 dtype，float64 输入会在 torch.full 处直接溢出报错，本题把掩码
常量按注意力张量自己的 dtype 生成，观测结果（被掩位置概率为 0）不变。
"""

import math

import torch
import torch.nn.functional as F


def _i(v, name, lo):
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise ValueError(f"{name} 必须是 >= {lo} 的 int")


def _num(v, name, lo, hi):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
        raise ValueError(f"{name} 必须在 [{lo}, {hi}]")


def _ema(x, dim, alpha):
    cur = x.select(dim, 0).unsqueeze(dim)
    for i in range(1, x.size(dim)):
        cur = alpha * x.select(dim, i).unsqueeze(dim) + (1.0 - alpha) * cur
    return cur.squeeze(dim)


def solve(query, key, value, wo, window_size=32, max_capacity_prompt=4096,
          kernel_size=5, pooling="avgpool", metric_mode="after", use_wo=True,
          use_norm=False, use_ema=True, alpha=0.3, tau=1.0):
    for t in (query, key, value):
        if not torch.is_tensor(t) or t.dim() != 4 or t.dtype not in (torch.float32, torch.float64):
            raise ValueError("query/key/value 必须是 float32/float64 的四维张量")
        if t.shape != query.shape or t.dtype != query.dtype:
            raise ValueError("query/key/value 的形状与 dtype 必须一致")
    if not torch.is_tensor(wo) or wo.dim() != 2 or wo.dtype != query.dtype:
        raise ValueError("wo 必须是与输入同 dtype 的二维张量")
    b, h, s, d = query.shape
    if wo.shape[1] != h * d:
        raise ValueError("wo 的第二维必须等于 H * D")
    _i(window_size, "window_size", 1)
    _i(max_capacity_prompt, "max_capacity_prompt", 2)
    _i(kernel_size, "kernel_size", 1)
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size 必须是奇数，否则池化会改变长度")
    if not (window_size < max_capacity_prompt and window_size < s):
        raise ValueError("需要 window_size < max_capacity_prompt 且 window_size < S")
    if pooling not in ("avgpool", "maxpool", "none"):
        raise ValueError("pooling 只能是 avgpool / maxpool / none")
    if metric_mode not in ("before", "after"):
        raise ValueError("metric_mode 只能是 before / after")
    for flag in (use_wo, use_norm, use_ema):
        if not isinstance(flag, bool):
            raise ValueError("use_wo / use_norm / use_ema 必须是布尔")
    _num(alpha, "alpha", 0.0, 1.0)
    _num(tau, "tau", 0.0, float("inf"))

    w, k = window_size, max_capacity_prompt - window_size
    aw = torch.matmul(query[..., -w:, :], key.transpose(2, 3)) / math.sqrt(d)
    corner = torch.full((w, w), torch.finfo(aw.dtype).min, dtype=aw.dtype, device=aw.device)
    c = torch.arange(w, device=aw.device)
    corner.masked_fill_(c < (c + 1).view(w, 1), 0)
    aw = aw.clone()
    aw[:, :, -w:, -w:] += corner
    aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(query.dtype)

    proj = wo.reshape(-1, h, d)
    past = aw[:, :, -w:, :-w]
    v_past = value[:, :, :-w]
    if metric_mode == "before":
        outs = torch.matmul(aw, value)
        sc = aw / (1.0 - aw)
        if use_wo:
            po, pv = (torch.einsum("bhwd,chd->bhwc", outs, proj),
                      torch.einsum("bhld,chd->bhlc", value, proj))
            diff = torch.norm(po.unsqueeze(3) - pv.unsqueeze(2), p=2, dim=-1)
            denom = torch.norm(po, p=2, dim=-1).unsqueeze(-1)
        else:
            diff = torch.norm(outs.unsqueeze(3) - value.unsqueeze(2), p=2, dim=-1)
            denom = torch.norm(outs, p=2, dim=-1).unsqueeze(-1)
        sc = sc * diff
        if use_norm:
            sc = sc / denom
        sc = sc[:, :, -w:, :-w]
        score = _ema(sc, -2, alpha) if use_ema else sc.mean(-2)
    else:
        mean = _ema(past, -2, alpha) if use_ema else past.mean(-2)
        outs_mean = torch.einsum("bhl,bhld->bhd", mean, v_past)
        score = mean / (1.0 - mean)
        if use_wo:
            po = torch.einsum("bhd,chd->bhc", outs_mean, proj)
            pv = torch.einsum("bhld,chd->bhlc", value, proj)[:, :, :-w]
            score = score * torch.norm(po.unsqueeze(2) - pv, p=2, dim=-1).pow(tau)
        else:
            score = score * torch.norm(outs_mean.unsqueeze(2) - v_past, p=2, dim=-1)

    if pooling == "avgpool":
        pooled = F.avg_pool1d(score, kernel_size, stride=1, padding=kernel_size // 2)
    elif pooling == "maxpool":
        pooled = F.max_pool1d(score, kernel_size, stride=1, padding=kernel_size // 2)
    else:
        pooled = score

    if s >= max_capacity_prompt:
        sel = torch.argsort(-pooled, dim=-1, stable=True)[..., :k]
        win = torch.arange(s - w, s, device=query.device).expand(b, h, w)
    else:
        sel = torch.zeros(b, h, 0, dtype=torch.int64, device=query.device)
        win = torch.arange(s, device=query.device).expand(b, h, s)
    return {"attn_weights": aw, "important_score": score, "pooled_score": pooled,
            "selected_indices": sel, "kept_positions": torch.cat([sel, win], dim=-1)}
