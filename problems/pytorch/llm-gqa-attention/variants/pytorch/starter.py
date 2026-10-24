import torch


def solve(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor,
    o_weight: torch.Tensor,
) -> torch.Tensor:
    """GQA 因果注意力：投影→拆头→RoPE→repeat_kv→缩放点积+因果掩码+softmax→合并→o_proj。

    hidden_states : (B, S, hidden)
    cos, sin      : (B, S, head_dim)  已算好的 RoPE
    q_weight:(Hq*d, hidden) q_bias:(Hq*d)  k/v_weight:(Hk*d, hidden) k/v_bias:(Hk*d)
    o_weight:(hidden, Hq*d) 无 o_bias（Qwen2 的 q/k/v 带 bias）
    head_dim=cos.shape[-1]，Hq=q_weight.shape[0]//head_dim，Hk=k_weight.shape[0]//head_dim。
    返回 attn_output (B, S, hidden)。softmax 用 float32、因果掩码上三角置 -inf、缩放 head_dim**-0.5。
    """
    # TODO: 实现 GQA 因果注意力（记得 repeat_kv 与因果掩码）
    raise NotImplementedError
