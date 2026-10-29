import numpy as np
import torch


def _close(a, b, tol=1e-5):
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    return torch.allclose(a, b, rtol=tol, atol=tol * max(1.0, float(a.abs().max())))


def _rms(x, weight, eps):
    mean_square = (x * x).mean(axis=-1, keepdims=True)
    return x / np.sqrt(mean_square + eps) * (1.0 + np.asarray(weight, dtype=np.float64))


def _oracle(
    hidden_states,
    cos,
    sin,
    ws,
    layer_type="sliding_attention",
    sliding_window=4096,
    query_pre_attn_scalar=256.0,
    softcap=50.0,
    rms_eps=1e-6,
    norm_target="output",
    mask_first=False,
):
    """独立 numpy 重算整层；norm_target / mask_first 用来构造两类常见错序对照。"""
    x = np.asarray(hidden_states, dtype=np.float64)
    batch, seq, hidden = x.shape
    cos = np.asarray(cos, dtype=np.float64)
    sin = np.asarray(sin, dtype=np.float64)
    head_dim = cos.shape[-1]
    n_heads = ws["q_weight"].shape[0] // head_dim
    n_kv = ws["k_weight"].shape[0] // head_dim

    def rms(t, w):
        return _rms(t, w, rms_eps)

    def split(t, heads):
        return t.reshape(batch, seq, heads, head_dim).transpose(0, 2, 1, 3)

    def rotate_half(t):
        half = head_dim // 2
        return np.concatenate((-t[..., half:], t[..., :half]), axis=-1)

    def attention(h):
        q = split(h @ ws["q_weight"].T, n_heads)
        k = split(h @ ws["k_weight"].T, n_kv)
        v = split(h @ ws["v_weight"].T, n_kv)
        q = q * cos[:, None] + rotate_half(q) * sin[:, None]
        k = k * cos[:, None] + rotate_half(k) * sin[:, None]
        repeat = n_heads // n_kv
        k = np.repeat(k, repeat, axis=1)
        v = np.repeat(v, repeat, axis=1)

        logits = np.einsum("bhqd,bhkd->bhqk", q, k) * query_pre_attn_scalar**-0.5
        q_idx = np.arange(seq)[None, :, None]
        kv_idx = np.arange(seq)[None, None, :]
        allowed = kv_idx <= q_idx
        if layer_type == "sliding_attention":
            allowed = allowed & (kv_idx > q_idx - sliding_window)
        if mask_first:
            logits = np.where(allowed, logits, -np.inf)
            if softcap is not None:
                logits = np.where(allowed, softcap * np.tanh(logits / softcap), -softcap)
        else:
            if softcap is not None:
                logits = softcap * np.tanh(logits / softcap)
            logits = np.where(allowed, logits, -np.inf)

        shifted = logits - logits.max(axis=-1, keepdims=True)
        weights = np.exp(shifted)
        weights = weights / weights.sum(axis=-1, keepdims=True)
        out = np.einsum("bhqk,bhkd->bhqd", weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq, n_heads * head_dim)
        return out @ ws["o_weight"].T, weights

    def mlp(normed):
        gate = normed @ ws["gate_weight"].T
        up = normed @ ws["up_weight"].T
        act = 0.5 * gate * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (gate + 0.044715 * gate**3)))
        return (act * up) @ ws["down_weight"].T

    residual = x
    attn, weights = attention(rms(x, ws["input_ln_weight"]))
    if norm_target == "output":
        x = residual + rms(attn, ws["post_attention_ln_weight"])
    else:
        x = rms(residual, ws["post_attention_ln_weight"]) + attn

    residual = x
    ff = mlp(rms(x, ws["pre_feedforward_ln_weight"]))
    if norm_target == "output":
        x = residual + rms(ff, ws["post_feedforward_ln_weight"])
    else:
        x = rms(residual, ws["post_feedforward_ln_weight"]) + ff
    return x, weights


def _weights(seed, hidden, n_heads, n_kv, head_dim, inter, dtype=torch.float64):
    gen = torch.Generator().manual_seed(seed)
    return {
        "input_ln_weight": torch.randn(hidden, generator=gen, dtype=dtype) * 0.1,
        "post_attention_ln_weight": torch.randn(hidden, generator=gen, dtype=dtype) * 0.1,
        "pre_feedforward_ln_weight": torch.randn(hidden, generator=gen, dtype=dtype) * 0.1,
        "post_feedforward_ln_weight": torch.randn(hidden, generator=gen, dtype=dtype) * 0.1,
        "q_weight": torch.randn(n_heads * head_dim, hidden, generator=gen, dtype=dtype) * 0.5,
        "k_weight": torch.randn(n_kv * head_dim, hidden, generator=gen, dtype=dtype) * 0.5,
        "v_weight": torch.randn(n_kv * head_dim, hidden, generator=gen, dtype=dtype) * 0.5,
        "o_weight": torch.randn(hidden, n_heads * head_dim, generator=gen, dtype=dtype) * 0.5,
        "gate_weight": torch.randn(inter, hidden, generator=gen, dtype=dtype) * 0.5,
        "up_weight": torch.randn(inter, hidden, generator=gen, dtype=dtype) * 0.5,
        "down_weight": torch.randn(hidden, inter, generator=gen, dtype=dtype) * 0.5,
    }


def _rope(seed, batch, seq, head_dim):
    gen = torch.Generator().manual_seed(seed)
    cos = torch.randn(batch, seq, head_dim, generator=gen, dtype=torch.float64)
    sin = torch.randn(batch, seq, head_dim, generator=gen, dtype=torch.float64)
    norm = (cos**2 + sin**2).sqrt()
    return cos / norm, sin / norm


def _np(ws):
    return {k: v.numpy() for k, v in ws.items()}


def check_example(solve):
    eye = torch.eye(4, dtype=torch.float64)
    zero = torch.zeros(4, dtype=torch.float64)
    hidden_states = eye.unsqueeze(0)
    cos = torch.ones(1, 4, 2, dtype=torch.float64)
    sin = torch.zeros(1, 4, 2, dtype=torch.float64)

    out, weights = solve(
        hidden_states, cos, sin, zero, zero, zero, zero, eye, eye, eye, eye, eye, eye, eye,
        layer_type="sliding_attention", sliding_window=2,
        query_pre_attn_scalar=2, softcap=10.0, rms_eps=1e-6,
    )
    assert out.dtype == torch.float32 and out.shape == (1, 4, 4)
    assert weights.dtype == torch.float32 and weights.shape == (1, 2, 4, 4)

    rows = [[round(float(v), 6) for v in row] for row in out[0].tolist()]
    assert rows == [
        [4.999999, 0.0, 0.0, 0.0],
        [0.128880, 4.995966, 0.0, 0.0],
        [0.0, 1.114066, 4.758161, 0.0],
        [0.0, 0.0, 0.128880, 4.995966],
    ], rows

    head0 = [[round(float(v), 6) for v in row] for row in weights[0, 0].tolist()]
    assert head0 == [
        [1.0, 0.0, 0.0, 0.0],
        [0.059786, 0.940214, 0.0, 0.0],
        [0.0, 0.5, 0.5, 0.0],
        [0.0, 0.0, 0.5, 0.5],
    ], head0


check_example.example = True


def check_sliding_window_support(solve):
    for layer_type, window in (("sliding_attention", 3), ("sliding_attention", 1), ("full_attention", 3)):
        ws = _weights(0, 8, 2, 1, 4, 10)
        cos, sin = _rope(1, 2, 6, 4)
        out, weights = solve(
            torch.randn(2, 6, 8, generator=torch.Generator().manual_seed(1), dtype=torch.float64),
            cos, sin, **ws, layer_type=layer_type, sliding_window=window,
        )
        for i in range(6):
            for j in range(6):
                allowed = j <= i and (layer_type == "full_attention" or j > i - window)
                cell = weights[:, :, i, j]
                if allowed:
                    assert float(cell.min()) > 0.0, f"{layer_type}/{window}: ({i},{j}) 应可见"
                else:
                    assert float(cell.abs().max()) == 0.0, f"{layer_type}/{window}: ({i},{j}) 应被掩掉"
            assert abs(float(weights[:, :, i, :].sum()) - weights.shape[0] * weights.shape[1]) < 1e-4, (
                "每 (batch, head) 的权重和应为 1"
            )


def check_softcap_before_mask(solve):
    ws = _weights(2, 8, 2, 2, 4, 12)
    cos, sin = _rope(3, 2, 6, 4)
    hidden = torch.randn(2, 6, 8, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    # 小 softcap + 大 logits：掩码先做会让被掩位置停在 -softcap，softmax 拿到真实权重
    out, _ = solve(hidden, cos, sin, **ws, layer_type="sliding_attention",
                   sliding_window=2, query_pre_attn_scalar=4, softcap=0.5)
    want, _ = _oracle(hidden.numpy(), cos.numpy(), sin.numpy(), _np(ws), layer_type="sliding_attention",
                      sliding_window=2, query_pre_attn_scalar=4, softcap=0.5)
    wrong, _ = _oracle(hidden.numpy(), cos.numpy(), sin.numpy(), _np(ws), layer_type="sliding_attention",
                       sliding_window=2, query_pre_attn_scalar=4, softcap=0.5, mask_first=True)
    assert _close(out, want), "softcap 必须先于掩码"
    assert not _close(out, wrong, tol=1e-3), "先掩码后 softcap 会得到不同的结果，测试应能区分"


def check_norm_uses_one_plus_weight(solve):
    # S=1、全单位投影：输出可手算。post_attention 的 gain=[1,0] 意味着该维缩放 (1+w)=2
    eye = torch.eye(2, dtype=torch.float64)
    zero = torch.zeros(2, dtype=torch.float64)
    gain = torch.tensor([1.0, 0.0], dtype=torch.float64)
    hidden = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    cos = torch.ones(1, 1, 2, dtype=torch.float64)
    sin = torch.zeros(1, 1, 2, dtype=torch.float64)

    def run(post_attn_gain):
        out, weights = solve(hidden, cos, sin, zero, post_attn_gain, zero, zero,
                             eye, eye, eye, eye, eye, eye, eye,
                             layer_type="full_attention", query_pre_attn_scalar=2, softcap=None)
        assert abs(float(weights[0, 0, 0, 0]) - 1.0) < 1e-6, "单 key 的 softmax 必须为 1"
        return out

    # gain=0：(1+w)=1 → 手算 out = [√2 + √2, 0] / ... 见下方实测值
    # gain=[1,0]：(1+w)=[2,1] → post-attention 段放大一倍，out 从 3.828426 变到 5.242639
    plain = run(zero)
    boosted = run(gain)
    assert _close(plain, [[[3.828426, 0.0]]], tol=1e-5), plain.tolist()
    assert _close(boosted, [[[5.242639, 0.0]]], tol=1e-5), boosted.tolist()
    assert abs(float(boosted[0, 0, 0]) - float(plain[0, 0, 0])) > 1.0, (
        "若把 (1+w) 写成 w，gain=[1,0] 会退化成 gain=0，两者输出将完全相同"
    )


def check_post_norm_on_sublayer_output(solve):
    ws = _weights(5, 6, 2, 2, 2, 8)
    cos, sin = _rope(6, 1, 5, 2)
    hidden = torch.randn(1, 5, 6, generator=torch.Generator().manual_seed(7), dtype=torch.float64) * 3.0
    out, _ = solve(hidden, cos, sin, **ws, layer_type="sliding_attention", sliding_window=4,
                   query_pre_attn_scalar=3, softcap=4.0)
    want, _ = _oracle(hidden.numpy(), cos.numpy(), sin.numpy(), _np(ws), layer_type="sliding_attention",
                      sliding_window=4, query_pre_attn_scalar=3, softcap=4.0)
    on_residual, _ = _oracle(hidden.numpy(), cos.numpy(), sin.numpy(), _np(ws), layer_type="sliding_attention",
                             sliding_window=4, query_pre_attn_scalar=3, softcap=4.0,
                             norm_target="residual")
    assert _close(out, want), "post-norm 必须作用在子层输出上"
    assert not _close(out, on_residual, tol=1e-3), "把 post-norm 套在残差上是另一种数据流，应能区分"


def check_query_pre_attn_scalar(solve):
    ws = _weights(8, 6, 2, 1, 2, 8)
    cos, sin = _rope(9, 1, 5, 2)
    hidden = torch.randn(1, 5, 6, generator=torch.Generator().manual_seed(10), dtype=torch.float64)
    out, _ = solve(hidden, cos, sin, **ws, layer_type="full_attention",
                   query_pre_attn_scalar=8, softcap=None)
    want, _ = _oracle(hidden.numpy(), cos.numpy(), sin.numpy(), _np(ws), layer_type="full_attention",
                      query_pre_attn_scalar=8, softcap=None)
    by_head_dim, _ = _oracle(hidden.numpy(), cos.numpy(), sin.numpy(), _np(ws), layer_type="full_attention",
                             query_pre_attn_scalar=2, softcap=None)
    assert _close(out, want), "缩放必须取 query_pre_attn_scalar**-0.5"
    assert not _close(out, by_head_dim, tol=1e-3), "用 head_dim**-0.5 代替会得到不同结果"


def check_gqa_head_mapping(solve):
    ws = _weights(11, 8, 4, 2, 2, 12)
    cos, sin = _rope(12, 2, 5, 2)
    hidden = torch.randn(2, 5, 8, generator=torch.Generator().manual_seed(13), dtype=torch.float64)
    out, weights = solve(hidden, cos, sin, **ws, layer_type="sliding_attention",
                         sliding_window=3, query_pre_attn_scalar=2, softcap=6.0)
    want, want_w = _oracle(hidden.numpy(), cos.numpy(), sin.numpy(), _np(ws), layer_type="sliding_attention",
                           sliding_window=3, query_pre_attn_scalar=2, softcap=6.0)
    assert weights.shape == (2, 4, 5, 5)
    assert _close(out, want), "GQA 下 q 头按 h // n_rep 复用到 kv 头"
    assert _close(weights, want_w)


def check_against_oracle(solve):
    cases = [
        dict(hidden=8, heads=2, kv=1, head_dim=4, inter=12, layer_type="sliding_attention",
             window=3, qpas=3, softcap=2.0, seq=5),
        dict(hidden=8, heads=4, kv=2, head_dim=2, inter=10, layer_type="sliding_attention",
             window=4, qpas=8, softcap=50.0, seq=6),
        dict(hidden=6, heads=3, kv=3, head_dim=2, inter=7, layer_type="full_attention",
             window=100, qpas=2, softcap=None, seq=4),
        dict(hidden=6, heads=3, kv=1, head_dim=2, inter=7, layer_type="full_attention",
             window=2, qpas=16, softcap=5.0, seq=3),
        dict(hidden=4, heads=1, kv=1, head_dim=4, inter=6, layer_type="sliding_attention",
             window=1, qpas=4, softcap=0.8, seq=5),
    ]
    for index, cfg in enumerate(cases):
        ws = _weights(20 + index, cfg["hidden"], cfg["heads"], cfg["kv"], cfg["head_dim"], cfg["inter"])
        cos, sin = _rope(30 + index, 2, cfg["seq"], cfg["head_dim"])
        hidden = torch.randn(2, cfg["seq"], cfg["hidden"], generator=torch.Generator().manual_seed(40 + index), dtype=torch.float64)
        out, weights = solve(hidden, cos, sin, **ws, layer_type=cfg["layer_type"],
                             sliding_window=cfg["window"], query_pre_attn_scalar=cfg["qpas"],
                             softcap=cfg["softcap"], rms_eps=1e-6)
        want, want_w = _oracle(hidden.numpy(), cos.numpy(), sin.numpy(), _np(ws),
                               layer_type=cfg["layer_type"], sliding_window=cfg["window"],
                               query_pre_attn_scalar=cfg["qpas"], softcap=cfg["softcap"], rms_eps=1e-6)
        assert _close(out, want), (index, cfg)
        assert _close(weights, want_w), (index, cfg)

    for eps in (1e-8, 1e-3):
        ws = _weights(50, 6, 2, 2, 2, 8)
        cos, sin = _rope(51, 1, 4, 2)
        hidden = torch.randn(1, 4, 6, generator=torch.Generator().manual_seed(52), dtype=torch.float64)
        out, _ = solve(hidden, cos, sin, **ws, layer_type="full_attention",
                       query_pre_attn_scalar=2, softcap=3.0, rms_eps=eps)
        want, _ = _oracle(hidden.numpy(), cos.numpy(), sin.numpy(), _np(ws), layer_type="full_attention",
                          query_pre_attn_scalar=2, softcap=3.0, rms_eps=eps)
        assert _close(out, want), f"eps={eps}"


def _hf_layer_output(hidden, cos, sin, ws, layer_type, window, qpas, softcap, eps):
    import transformers as tr
    from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer

    hidden = hidden.float()
    cos, sin = cos.float(), sin.float()
    batch, seq, dim = hidden.shape
    head_dim = cos.shape[-1]
    n_heads = ws["q_weight"].shape[0] // head_dim
    n_kv = ws["k_weight"].shape[0] // head_dim
    inter = ws["gate_weight"].shape[0]
    cfg = tr.Gemma2Config(
        hidden_size=dim,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        head_dim=head_dim,
        intermediate_size=inter,
        attn_logit_softcapping=softcap,
        query_pre_attn_scalar=qpas,
        sliding_window=window,
        rms_norm_eps=eps,
        attention_bias=False,
        layer_types=[layer_type],
        num_hidden_layers=1,
        vocab_size=128,
    )
    layer = Gemma2DecoderLayer(cfg, 0)
    with torch.no_grad():
        layer.self_attn.q_proj.weight.copy_(ws["q_weight"].float())
        layer.self_attn.k_proj.weight.copy_(ws["k_weight"].float())
        layer.self_attn.v_proj.weight.copy_(ws["v_weight"].float())
        layer.self_attn.o_proj.weight.copy_(ws["o_weight"].float())
        layer.mlp.gate_proj.weight.copy_(ws["gate_weight"].float())
        layer.mlp.up_proj.weight.copy_(ws["up_weight"].float())
        layer.mlp.down_proj.weight.copy_(ws["down_weight"].float())
        layer.input_layernorm.weight.copy_(ws["input_ln_weight"].float())
        layer.post_attention_layernorm.weight.copy_(ws["post_attention_ln_weight"].float())
        layer.pre_feedforward_layernorm.weight.copy_(ws["pre_feedforward_ln_weight"].float())
        layer.post_feedforward_layernorm.weight.copy_(ws["post_feedforward_ln_weight"].float())

        q_idx = torch.arange(seq).view(1, seq, 1)
        kv_idx = torch.arange(seq).view(1, 1, seq)
        allowed = kv_idx <= q_idx
        if layer_type == "sliding_attention":
            allowed = allowed & (kv_idx > q_idx - window)
        mask = torch.zeros(batch, 1, seq, seq).masked_fill(~allowed, torch.finfo(torch.float32).min)
        return layer(hidden, position_embeddings=(cos, sin), attention_mask=mask)


def check_against_transformers(solve):
    for index, (hidden_dim, heads, kv, head_dim, inter, layer_type, window, qpas, softcap, seq) in enumerate(
        (
            (8, 2, 1, 4, 12, "sliding_attention", 3, 3, 2.0, 5),
            (8, 4, 2, 2, 10, "sliding_attention", 4, 8, 50.0, 6),
            (6, 3, 3, 2, 7, "full_attention", 100, 2, None, 4),
            (6, 3, 1, 2, 7, "full_attention", 2, 16, 5.0, 3),
        )
    ):
        ws = _weights(60 + index, hidden_dim, heads, kv, head_dim, inter)
        cos, sin = _rope(70 + index, 2, seq, head_dim)
        hidden = torch.randn(2, seq, hidden_dim, generator=torch.Generator().manual_seed(80 + index), dtype=torch.float64)
        out, _ = solve(hidden, cos, sin, **ws, layer_type=layer_type, sliding_window=window,
                       query_pre_attn_scalar=qpas, softcap=softcap)
        ref = _hf_layer_output(hidden, cos, sin, ws, layer_type, window, qpas, softcap, 1e-6)
        assert _close(out, ref, tol=2e-5), (index, layer_type, softcap)


def check_invalid(solve):
    eye = torch.eye(4, dtype=torch.float64)
    zero = torch.zeros(4, dtype=torch.float64)
    hidden = torch.zeros(1, 3, 4, dtype=torch.float64)
    cos = torch.ones(1, 3, 2, dtype=torch.float64)
    sin = torch.zeros(1, 3, 2, dtype=torch.float64)
    eye2 = torch.eye(2, dtype=torch.float64)

    for bad in (
        lambda: solve(hidden[0], cos, sin, zero, zero, zero, zero, eye, eye, eye, eye, eye, eye, eye),
        lambda: solve(hidden, cos[0], sin, zero, zero, zero, zero, eye, eye, eye, eye, eye, eye, eye),
        lambda: solve(hidden, cos, sin[:, :2], zero, zero, zero, zero, eye, eye, eye, eye, eye, eye, eye),
        lambda: solve(hidden, cos, sin, zero[:3], zero, zero, zero, eye, eye, eye, eye, eye, eye, eye),
        lambda: solve(hidden, cos, sin, zero, zero, zero, zero, eye2, eye, eye, eye, eye, eye, eye),
        lambda: solve(hidden, cos, sin, zero, zero, zero, zero, eye, eye2, eye, eye, eye, eye, eye),
        lambda: solve(hidden, cos, sin, zero, zero, zero, zero, eye, eye, eye, eye[:2], eye, eye, eye),
        lambda: solve(hidden, cos, sin, zero, zero, zero, zero, eye, eye, eye, eye, eye[:2], eye, eye),
        lambda: solve(hidden, cos, sin, zero, zero, zero, zero, eye, eye, eye, eye, eye, eye, eye2),
        lambda: solve(hidden, cos, sin, zero, zero, zero, zero, eye, eye, eye, eye, eye, eye, eye,
                      layer_type="global"),
        lambda: solve(hidden, cos, sin, zero, zero, zero, zero, eye, eye, eye, eye, eye, eye, eye,
                      sliding_window=0),
        lambda: solve(hidden, cos, sin, zero, zero, zero, zero, eye, eye, eye, eye, eye, eye, eye,
                      sliding_window=2.0),
        lambda: solve(hidden, cos, sin, zero, zero, zero, zero, eye, eye, eye, eye, eye, eye, eye,
                      query_pre_attn_scalar=0),
        lambda: solve(hidden, cos, sin, zero, zero, zero, zero, eye, eye, eye, eye, eye, eye, eye,
                      query_pre_attn_scalar=float("nan")),
        lambda: solve(hidden, cos, sin, zero, zero, zero, zero, eye, eye, eye, eye, eye, eye, eye,
                      softcap=0.0),
        lambda: solve(hidden, cos, sin, zero, zero, zero, zero, eye, eye, eye, eye, eye, eye, eye,
                      softcap=-1.0),
        lambda: solve(hidden, cos, sin, zero, zero, zero, zero, eye, eye, eye, eye, eye, eye, eye,
                      rms_eps=0.0),
        lambda: solve(hidden, cos, sin, zero, zero, zero, zero, eye, eye, eye, eye, eye, eye, eye,
                      rms_eps=float("inf")),
        lambda: solve(hidden, torch.ones(1, 3, 3, dtype=torch.float64), sin, zero, zero, zero, zero,
                      eye, eye, eye, eye, eye, eye, eye),
        lambda: solve([[0.0]], cos, sin, zero, zero, zero, zero, eye, eye, eye, eye, eye, eye, eye),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
