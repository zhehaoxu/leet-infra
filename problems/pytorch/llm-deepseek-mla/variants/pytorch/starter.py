import torch


def solve(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_a_proj_weight: torch.Tensor,
    q_a_norm_weight: torch.Tensor,
    q_b_proj_weight: torch.Tensor,
    kv_a_proj_weight: torch.Tensor,
    kv_a_norm_weight: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    n_heads: int,
    nope_head_dim: int,
    rope_head_dim: int,
    v_head_dim: int,
    rms_eps: float = 1e-6,
) -> torch.Tensor:
    """DeepSeek MLA：低秩潜在压缩注意力 + 解耦 RoPE。

    hidden_states : (B, S, hidden)
    cos, sin      : (B, S, rope_head_dim)  已算好的 RoPE（仅作用于 rope 分量）
    q_a_proj_weight (d_q, hidden), q_a_norm_weight (d_q,), q_b_proj_weight (n_heads*d_qk, d_q)
    kv_a_proj_weight (d_kv+rope_head_dim, hidden), kv_a_norm_weight (d_kv,),
    kv_b_proj_weight (n_heads*(nope+v), d_kv), o_proj_weight (hidden, n_heads*v)
    d_q=q_a_proj_weight.shape[0], d_kv=kv_a_norm_weight.shape[0], d_qk=nope+rope, rms_eps 默认 1e-6。

    线性层 [out,in] 布局、F.linear(x,W)=x@Wᵀ；RMSNorm fp32 内算 rsqrt(mean(x²)+eps) 再乘 gain。
    步骤：q_latent→q(split nope/rot)；compressed→c_kv(RMSNorm)+k_pe；仅 q_rot/k_pe 加 RoPE；
    c_kv→kv_b_proj 展开每头 k_nope,v；k=cat(k_nope,k_pe广播)；score=QKᵀ/√d_qk+因果-inf→fp32 softmax→AV→o_proj。
    num_key_value_heads==n_heads，无需 repeat_kv。返回 (B, S, hidden)。
    """
    # TODO: 潜在压缩 → 解耦 RoPE → 展开 k/v → 因果注意力 → o_proj
    raise NotImplementedError
