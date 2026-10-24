import torch


def solve(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    head_dim: int,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE：按 position_ids 生成 cos/sin，对 q、k 施加旋转位置编码。

    q : (B, Hq, S, head_dim)    k : (B, Hk, S, head_dim)
    position_ids : (B, S) 整数   head_dim : 偶数   base : RoPE 基频
    返回 (q_rot, k_rot)，形状与输入一致。inv_freq=base**(-2i/d)，cos/sin 用 float32。
    """
    # TODO: 生成 inv_freq/emb/cos/sin，实现 rotate_half 并施加 q*cos + rotate_half(q)*sin
    raise NotImplementedError
