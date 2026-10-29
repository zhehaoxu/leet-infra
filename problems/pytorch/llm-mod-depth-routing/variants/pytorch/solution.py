import math

import torch

_FLOAT_DTYPES = (torch.float32, torch.float64)


def _check_float_tensor(name, value, ndim):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.dtype not in _FLOAT_DTYPES:
        raise ValueError(f"{name} must be float32 or float64")
    if value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _check_capacities(capacities, layers, tokens):
    if isinstance(capacities, bool) or not isinstance(capacities, (list, tuple)):
        raise ValueError("capacities must be a list or tuple of ints")
    if len(capacities) != layers:
        raise ValueError("capacities must hold one entry per layer")
    out = []
    for value in capacities:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("each capacity must be an int")
        if value < 0:
            raise ValueError("capacity must be >= 0")
        out.append(min(value, tokens))
    return out


def solve(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
    block_weights: torch.Tensor,
    capacities: list[int] | tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mixture-of-Depths 前向：逐层按 router 分数取容量，只让选中的 token 过 block。

    hidden_states : (T, d)，T 个 token 的当前状态（初始输入）
    router_weights : (L, d)，第 l 层 router 向量 w_l
    block_weights : (L, d, d)，第 l 层 block 的唯一投影矩阵 W_l（自注意力共享 q/k/v 投影）
    capacities : 长度 L 的整数序列，第 l 层最多路由 C_l 个 token；超过 T 按 T 处理

    逐层口径（内部 float64）：
      1. 分数 s_i = x_i · w_l（用**当前**状态算，不是初始输入）
      2. 选容量 k_l = min(C_l, T)：按 (s_i 降序, token 下标升序) 取前 k_l 个，
         取到的下标再按**原始下标升序**排成子序列
      3. 子序列 X̃ 过 block：H = X̃ W_l；A = H Hᵀ / sqrt(d)；因果掩码（j <= i）；
         P = softmax(A)；F = P H
      4. 选中 token：x_i ← x_i + s_i · F_row；未选中 token 严格保持原值
    返回 (output, route_mask, router_scores)：
      output (T, d) float32；route_mask (L, T) bool（True = 该层被路由）；
      router_scores (L, T) float32（各层路由决策时用到的分数）。
    """
    _check_float_tensor("hidden_states", hidden_states, 2)
    tokens, dim = hidden_states.shape
    if tokens < 1 or dim < 1:
        raise ValueError("hidden_states must have shape (T >= 1, d >= 1)")
    _check_float_tensor("router_weights", router_weights, 2)
    _check_float_tensor("block_weights", block_weights, 3)
    layers = router_weights.shape[0]
    if layers < 1:
        raise ValueError("at least one layer is required")
    if router_weights.shape[1] != dim:
        raise ValueError("router_weights must have shape (L, d)")
    if tuple(block_weights.shape) != (layers, dim, dim):
        raise ValueError("block_weights must have shape (L, d, d)")
    caps = _check_capacities(capacities, layers, tokens)

    x = hidden_states.to(torch.float64)
    routers = router_weights.to(torch.float64)
    blocks = block_weights.to(torch.float64)
    scale = 1.0 / math.sqrt(dim)
    route_mask = torch.zeros(layers, tokens, dtype=torch.bool)
    router_scores = torch.zeros(layers, tokens, dtype=torch.float64)

    for layer in range(layers):
        scores = x @ routers[layer]
        router_scores[layer] = scores
        capacity = caps[layer]
        if capacity == 0:
            continue
        order = torch.argsort(scores, descending=True, stable=True)
        chosen = order[:capacity].sort().values
        route_mask[layer, chosen] = True
        subset = x[chosen]
        h = subset @ blocks[layer]
        size = h.shape[0]
        attn = (h @ h.transpose(0, 1)) * scale
        attn = attn.masked_fill(
            ~torch.tril(torch.ones(size, size, dtype=torch.bool)), float("-inf")
        )
        mixed = torch.softmax(attn, dim=-1) @ h
        rows = subset + scores[chosen].unsqueeze(-1) * mixed
        x = x.index_copy(0, chosen, rows)

    return (
        x.to(torch.float32),
        route_mask,
        router_scores.to(torch.float32),
    )
