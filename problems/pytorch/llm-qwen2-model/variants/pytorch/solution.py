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


def _decoder_layer(x, cos, sin, causal, L, eps):
    B, S, _ = x.shape
    head_dim = cos.shape[-1]
    n_q = L["q_weight"].shape[0] // head_dim
    n_kv = L["k_weight"].shape[0] // head_dim

    residual = x
    h = _rmsnorm(x, L["input_ln_weight"], eps)
    q = F.linear(h, L["q_weight"], L["q_bias"]).view(B, S, n_q, head_dim).transpose(1, 2)
    k = F.linear(h, L["k_weight"], L["k_bias"]).view(B, S, n_kv, head_dim).transpose(1, 2)
    v = F.linear(h, L["v_weight"], L["v_bias"]).view(B, S, n_kv, head_dim).transpose(1, 2)

    c = cos.unsqueeze(1)
    s = sin.unsqueeze(1)
    q = q * c + _rotate_half(q) * s
    k = k * c + _rotate_half(k) * s

    n_rep = n_q // n_kv
    k = _repeat_kv(k, n_rep)
    v = _repeat_kv(v, n_rep)

    attn = (q @ k.transpose(2, 3)) * (head_dim ** -0.5)
    attn = attn + causal
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    o = (attn @ v).transpose(1, 2).reshape(B, S, n_q * head_dim)
    o = F.linear(o, L["o_weight"])
    h = residual + o

    residual = h
    h = _rmsnorm(h, L["post_attention_ln_weight"], eps)
    h = F.linear(F.silu(F.linear(h, L["gate_weight"])) * F.linear(h, L["up_weight"]), L["down_weight"])
    return residual + h


def solve(
    input_ids: torch.Tensor,
    embed_weight: torch.Tensor,
    final_norm_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    layers: list,
    head_dim: int,
    base: float = 10000.0,
    rms_eps: float = 1e-6,
) -> torch.Tensor:
    B, S = input_ids.shape

    # 1) token 嵌入
    x = F.embedding(input_ids, embed_weight)

    # 2) RoPE 现场生成（位置 0..S-1）
    pos = torch.arange(S, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = pos[:, None] * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0).expand(B, S, head_dim)
    sin = emb.sin().unsqueeze(0).expand(B, S, head_dim)

    # 3) 逐层解码
    causal = torch.full((S, S), float("-inf"), device=x.device).triu(1)
    for L in layers:
        x = _decoder_layer(x, cos, sin, causal, L, rms_eps)

    # 4) 顶层 norm + lm_head
    x = _rmsnorm(x, final_norm_weight, rms_eps)
    return F.linear(x, lm_head_weight)
