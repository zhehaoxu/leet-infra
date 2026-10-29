import math

import torch

_LAYER_TYPES = ("sliding_attention", "full_attention")


def _tensor(value, name, dims, rows=None):
    if not torch.is_tensor(value):
        raise ValueError(f"{name} 必须是张量")
    if value.dim() != dims:
        raise ValueError(f"{name} 必须是 {dims} 维张量")
    if rows is not None and value.shape[0] != rows:
        raise ValueError(f"{name} 的行数必须是 {rows}")
    return value


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是实数")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} 必须是有限正数")
    return value


def _rms_norm(x, weight, eps):
    x = x.to(torch.float64)
    normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return normed * (1.0 + weight.to(torch.float64))


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def solve(
    hidden_states,
    cos,
    sin,
    input_ln_weight,
    post_attention_ln_weight,
    pre_feedforward_ln_weight,
    post_feedforward_ln_weight,
    q_weight,
    k_weight,
    v_weight,
    o_weight,
    gate_weight,
    up_weight,
    down_weight,
    layer_type="sliding_attention",
    sliding_window=4096,
    query_pre_attn_scalar=256.0,
    softcap=50.0,
    rms_eps=1e-6,
):
    _tensor(hidden_states, "hidden_states", 3)
    _tensor(cos, "cos", 3)
    _tensor(sin, "sin", 3)
    batch, seq, hidden = hidden_states.shape
    if cos.shape != sin.shape:
        raise ValueError("cos 与 sin 形状必须一致")
    if cos.shape[:2] != (batch, seq):
        raise ValueError("cos/sin 必须是 (B, S, head_dim)")
    head_dim = cos.shape[-1]
    if head_dim % 2 or head_dim <= 0:
        raise ValueError("head_dim 必须是正偶数")
    if layer_type not in _LAYER_TYPES:
        raise ValueError(f"未知的 layer_type: {layer_type!r}")
    if isinstance(sliding_window, bool) or not isinstance(sliding_window, int) or sliding_window < 1:
        raise ValueError("sliding_window 必须是 >= 1 的整数")
    query_pre_attn_scalar = _positive(query_pre_attn_scalar, "query_pre_attn_scalar")
    rms_eps = _positive(rms_eps, "rms_eps")
    if softcap is not None:
        softcap = _positive(softcap, "softcap")

    for name, weight in (
        ("input_ln_weight", input_ln_weight),
        ("post_attention_ln_weight", post_attention_ln_weight),
        ("pre_feedforward_ln_weight", pre_feedforward_ln_weight),
        ("post_feedforward_ln_weight", post_feedforward_ln_weight),
    ):
        _tensor(weight, name, 1, hidden)

    for name, weight in (
        ("q_weight", q_weight),
        ("k_weight", k_weight),
        ("v_weight", v_weight),
        ("gate_weight", gate_weight),
        ("up_weight", up_weight),
    ):
        _tensor(weight, name, 2)
        if weight.shape[1] != hidden:
            raise ValueError(f"{name} 的输入维必须是 {hidden}")

    _tensor(o_weight, "o_weight", 2)
    _tensor(down_weight, "down_weight", 2)
    if down_weight.shape[0] != hidden:
        raise ValueError(f"down_weight 的输出维必须是 {hidden}")

    if q_weight.shape[0] % head_dim:
        raise ValueError("q_weight 的输出维必须是 head_dim 的整数倍")
    if k_weight.shape[0] % head_dim or v_weight.shape != k_weight.shape:
        raise ValueError("k_weight / v_weight 必须是 (n_kv_heads * head_dim, hidden)")
    n_heads = q_weight.shape[0] // head_dim
    n_kv_heads = k_weight.shape[0] // head_dim
    if n_heads % n_kv_heads:
        raise ValueError("query 头数必须能被 kv 头数整除")
    if o_weight.shape != (hidden, n_heads * head_dim):
        raise ValueError("o_weight 必须是 (hidden, n_heads * head_dim)")

    inter = gate_weight.shape[0]
    if up_weight.shape != (inter, hidden):
        raise ValueError("up_weight 必须与 gate_weight 同形")
    if gate_weight.shape[0] < 1:
        raise ValueError("MLP 中间维必须为正")
    if down_weight.shape != (hidden, inter):
        raise ValueError("down_weight 必须是 (hidden, intermediate)")

    x = hidden_states.to(torch.float64)
    cos64 = cos.to(torch.float64).unsqueeze(1)
    sin64 = sin.to(torch.float64).unsqueeze(1)

    residual = x
    normed = _rms_norm(x, input_ln_weight, rms_eps)

    q = torch.nn.functional.linear(normed, q_weight.to(torch.float64))
    k = torch.nn.functional.linear(normed, k_weight.to(torch.float64))
    v = torch.nn.functional.linear(normed, v_weight.to(torch.float64))
    q = q.view(batch, seq, n_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq, n_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch, seq, n_kv_heads, head_dim).transpose(1, 2)

    q = q * cos64 + _rotate_half(q) * sin64
    k = k * cos64 + _rotate_half(k) * sin64

    if n_kv_heads != n_heads:
        repeat = n_heads // n_kv_heads
        k = k[:, :, None, :, :].expand(batch, n_kv_heads, repeat, seq, head_dim).reshape(
            batch, n_heads, seq, head_dim
        )
        v = v[:, :, None, :, :].expand(batch, n_kv_heads, repeat, seq, head_dim).reshape(
            batch, n_heads, seq, head_dim
        )

    logits = torch.matmul(q, k.transpose(2, 3)) * (query_pre_attn_scalar**-0.5)
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap

    q_idx = torch.arange(seq, dtype=torch.int64).view(1, seq, 1)
    kv_idx = torch.arange(seq, dtype=torch.int64).view(1, 1, seq)
    allowed = kv_idx <= q_idx
    if layer_type == "sliding_attention":
        allowed = allowed & (kv_idx > q_idx - sliding_window)
    logits = logits.masked_fill(~allowed, float("-inf"))

    weights = torch.softmax(logits, dim=-1)
    attn = torch.matmul(weights, v).transpose(1, 2).reshape(batch, seq, n_heads * head_dim)
    attn = torch.nn.functional.linear(attn, o_weight.to(torch.float64))
    attn = _rms_norm(attn, post_attention_ln_weight, rms_eps)
    x = residual + attn

    residual = x
    normed = _rms_norm(x, pre_feedforward_ln_weight, rms_eps)
    gate = torch.nn.functional.linear(normed, gate_weight.to(torch.float64))
    up = torch.nn.functional.linear(normed, up_weight.to(torch.float64))
    mlp = torch.nn.functional.linear(
        torch.nn.functional.gelu(gate, approximate="tanh") * up,
        down_weight.to(torch.float64),
    )
    mlp = _rms_norm(mlp, post_feedforward_ln_weight, rms_eps)
    x = residual + mlp

    return x.to(torch.float32), weights.to(torch.float32)
