import torch


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

    非法输入抛 ValueError：张量非张量/非浮点/维数不对/含 nan、±inf；
    router 或 block 的形状与 d、L 不匹配；capacities 不是 list/tuple、
    长度不等于 L、元素不是 int（含 bool/float）、或为负数。
    """
    # TODO: 逐层打分 → 容量选择 → 子序列因果自注意力 → 按分数缩放写回残差
    raise NotImplementedError
