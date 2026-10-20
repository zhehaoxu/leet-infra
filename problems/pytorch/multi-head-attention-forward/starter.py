import torch


def solve(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
          num_heads: int, causal: bool = False) -> torch.Tensor:
    """Multi-head attention 前向。

    q/k/v: (B, L, D)，序列长度相同，按特征维连续切分为 num_heads 个头
    num_heads: 头数 H，要求 D % H == 0，每头维度 d = D // H
    causal: True 时位置 i 只能 attend 到 j <= i
    """
    # TODO: 1) 校验 D % num_heads，不整除抛 ValueError
    #       2) (B, L, D) -> (B, H, L, d)
    #       3) scores = Q @ K^T / sqrt(d)，可选 causal -inf mask，softmax 沿 key 维
    #       4) out = Attn @ V，拼回 (B, L, D)
    # 禁止使用 F.scaled_dot_product_attention
    raise NotImplementedError("请实现 solve()")
