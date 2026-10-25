"""参考答案：DeepSeek-V3 多头潜在注意力 (MLA)。对拍 HF DeepseekV3Attention(eager, rope_interleave=False)。"""

import torch
import torch.nn.functional as F


def _rotate_half(x):
    d = x.shape[-1]
    return torch.cat((-x[..., d // 2:], x[..., : d // 2]), dim=-1)


def _rmsnorm(x, weight, eps):
    dt = x.dtype
    x = x.to(torch.float32)
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return weight * x.to(dt)


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
    B, S, hidden = hidden_states.shape
    qk = nope_head_dim + rope_head_dim
    kv_lora = kv_a_norm_weight.shape[0]

    # 1) query 潜在压缩路径
    q_latent = _rmsnorm(F.linear(hidden_states, q_a_proj_weight), q_a_norm_weight, rms_eps)
    q_states = F.linear(q_latent, q_b_proj_weight).view(B, S, n_heads, qk).transpose(1, 2)
    q_pass, q_rot = torch.split(q_states, [nope_head_dim, rope_head_dim], dim=-1)

    # 2) KV 潜在压缩 + 解耦 rope 分量
    compressed = F.linear(hidden_states, kv_a_proj_weight)
    kv_nope, k_rot = torch.split(compressed, [kv_lora, rope_head_dim], dim=-1)
    kv_nope = _rmsnorm(kv_nope, kv_a_norm_weight, rms_eps)
    kv_nope = kv_nope.view(B, 1, S, kv_lora)
    k_rot = k_rot.view(B, 1, S, rope_head_dim)

    # 3) 仅对 rope 分量施加旋转位置编码
    c = cos.unsqueeze(1)
    s = sin.unsqueeze(1)
    q_rot = q_rot * c + _rotate_half(q_rot) * s
    k_rot = k_rot * c + _rotate_half(k_rot) * s

    query_states = torch.cat((q_pass, q_rot), dim=-1)  # (B,nh,S,qk)

    # 4) 潜在展开成每头 k/v（k_pe 广播到所有头）
    kv = F.linear(kv_nope, kv_b_proj_weight).view(B, S, -1, nope_head_dim + v_head_dim).transpose(1, 2)
    k_nope, value_states = torch.split(kv, [nope_head_dim, v_head_dim], dim=-1)
    k_rot = k_rot.expand(-1, k_nope.shape[1], -1, -1)
    key_states = torch.cat((k_nope, k_rot), dim=-1)  # (B,nh,S,qk)

    # 5) 缩放点积因果注意力（num_key_value_heads==n_heads，无需 repeat_kv）
    attn = (query_states @ key_states.transpose(2, 3)) * (qk ** -0.5)
    causal = torch.full((1, 1, S, S), float("-inf"), device=hidden_states.device).triu(1)
    attn = attn + causal
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(query_states.dtype)
    out = (attn @ value_states).transpose(1, 2).reshape(B, S, n_heads * v_head_dim)
    return F.linear(out, o_proj_weight)
