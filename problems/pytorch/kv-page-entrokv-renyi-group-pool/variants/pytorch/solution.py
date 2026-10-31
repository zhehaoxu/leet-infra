"""参考解：Page-EntroKV 的分组 Rényi 熵与页级保留（照抄 page_entrokv/core.py L17-L44）。

上游 PageEntroKVGroupPooler 在 core.py L6；__init__ L7-15；renyi_entropy L17-22
（alpha == 1.0 分支 L19-20，Rényi 分支 L21-22，eps = 1e-12）；forward L24-44：
sink 掩码 L27-28、隔离重归一化 L29-30、熵 L31、按原始 attn 分组 L32、组内 softmax L34、
池化 L35、零填充 L36-38、页 max L40、topk L41、掩码 L42-44。

上游 forward 只返回页掩码；entropies / group_weights / page_scores / retained_tokens
四个导出键是本题自己的契约。仓库的 baselines/ 与 kernels/ 均自标 PENDING，本题只照 core.py。
"""

import torch

INT_NAMES = ("num_retained_blocks", "num_query_heads", "num_kv_heads", "block_size", "sink_size")


def _check_real(name, v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{name} 必须是实数")
    if v <= 0:
        raise ValueError(f"{name} 必须是正数")


def solve(attn, num_retained_blocks, num_query_heads, num_kv_heads,
          block_size=16, tau_g=0.5, alpha=2.0, sink_size=4):
    for name, v in zip(INT_NAMES, (num_retained_blocks, num_query_heads, num_kv_heads,
                                   block_size, sink_size)):
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"{name} 必须是整数")
    if num_retained_blocks < 0:
        raise ValueError("num_retained_blocks 必须 >= 0")
    if sink_size < 0:
        raise ValueError("sink_size 必须 >= 0")
    for name, v in (("block_size", block_size),
                    ("num_query_heads", num_query_heads), ("num_kv_heads", num_kv_heads)):
        if v < 1:
            raise ValueError(f"{name} 必须是正整数")
    _check_real("tau_g", tau_g)
    _check_real("alpha", alpha)
    if not torch.is_tensor(attn) or attn.dim() != 3:
        raise ValueError("attn 必须是三维张量")
    if not attn.is_floating_point():
        raise ValueError("attn 必须是浮点张量")
    B, H_Q, T = attn.shape
    if B < 1 or H_Q < 1 or T < 1:
        raise ValueError("attn 的三个维度都必须 >= 1")
    if H_Q != num_query_heads:
        raise ValueError("num_query_heads 必须等于 attn.shape[1]")
    if H_Q % num_kv_heads != 0:
        raise ValueError("H_Q 必须能被 num_kv_heads 整除")
    if bool((attn < 0).any()):
        raise ValueError("attn 必须非负")
    r = H_Q // num_kv_heads
    nb = (T + block_size - 1) // block_size
    if num_retained_blocks > nb:
        raise ValueError("num_retained_blocks 超过页数")

    x = attn.to(torch.float64)
    sink_mask = torch.zeros(T, dtype=torch.bool, device=x.device)
    sink_mask[:min(sink_size, T)] = True
    nonsink = x.masked_fill(sink_mask.view(1, 1, T), 0.0)
    nonsink = nonsink / nonsink.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    if alpha == 1.0:
        ent = -(nonsink * torch.log(nonsink + 1e-12)).sum(dim=-1)
    else:
        ent = (1.0 / (1.0 - alpha)) * torch.log(
            (nonsink ** alpha).sum(dim=-1) + 1e-12)

    ent_g = ent.view(B, num_kv_heads, r)
    weights = torch.softmax(-ent_g / tau_g, dim=-1)
    pooled = (weights.unsqueeze(-1) * x.view(B, num_kv_heads, r, T)).sum(dim=2)
    pad = (block_size - T % block_size) % block_size
    if pad:
        pooled = torch.nn.functional.pad(pooled, (0, pad))
    nb = pooled.shape[-1] // block_size
    scores = pooled.view(B, num_kv_heads, nb, block_size).max(dim=-1).values

    order = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :num_retained_blocks]
    pages = torch.zeros_like(scores, dtype=torch.int64)
    pages.scatter_(-1, order, 1)
    tokens = pages.repeat_interleave(block_size, dim=-1)[..., :T]
    return {
        "entropies": ent,
        "group_weights": weights,
        "page_scores": scores,
        "retained_pages": pages,
        "retained_tokens": tokens,
    }
