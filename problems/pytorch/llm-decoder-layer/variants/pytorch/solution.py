import torch
import torch.nn.functional as F


def _rotate_half(x):
    d = x.shape[-1]
    return torch.cat((-x[..., d // 2:], x[..., : d // 2]), dim=-1)


def _repeat_kv(x, n_rep):
    B, H, S, D = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None].expand(B, H, n_rep, S, D).reshape(B, H * n_rep, S, D)


def _rmsnorm(x, weight, eps):
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(input_dtype)


def solve(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    input_ln_weight: torch.Tensor,
    post_attention_ln_weight: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor,
    o_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    rms_eps: float = 1e-6,
) -> torch.Tensor:
    B, S, _ = hidden_states.shape
    head_dim = cos.shape[-1]
    n_q = q_weight.shape[0] // head_dim
    n_kv = k_weight.shape[0] // head_dim

    # ---- pre-norm + self attention ----
    residual = hidden_states
    h = _rmsnorm(hidden_states, input_ln_weight, rms_eps)

    q = F.linear(h, q_weight, q_bias).view(B, S, n_q, head_dim).transpose(1, 2)
    k = F.linear(h, k_weight, k_bias).view(B, S, n_kv, head_dim).transpose(1, 2)
    v = F.linear(h, v_weight, v_bias).view(B, S, n_kv, head_dim).transpose(1, 2)

    c = cos.unsqueeze(1)
    s = sin.unsqueeze(1)
    q = q * c + _rotate_half(q) * s
    k = k * c + _rotate_half(k) * s

    n_rep = n_q // n_kv
    k = _repeat_kv(k, n_rep)
    v = _repeat_kv(v, n_rep)

    attn = (q @ k.transpose(2, 3)) * (head_dim ** -0.5)
    causal = torch.full((S, S), float("-inf"), device=attn.device).triu(1)
    attn = attn + causal
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    o = (attn @ v).transpose(1, 2).reshape(B, S, n_q * head_dim)
    o = F.linear(o, o_weight)
    h = residual + o

    # ---- pre-norm + SwiGLU MLP ----
    residual = h
    h = _rmsnorm(h, post_attention_ln_weight, rms_eps)
    h = F.linear(F.silu(F.linear(h, gate_weight)) * F.linear(h, up_weight), down_weight)
    return residual + h
