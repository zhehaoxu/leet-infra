import math

import torch


# ---------- 纯 Python oracle（float64，不依赖 torch 语义） ----------

def _silu(x):
    return x / (1.0 + math.exp(-x)) if x >= 0 else x * math.exp(x) / (1.0 + math.exp(x))


def _gelu_tanh(x):
    inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1.0 + math.tanh(inner))


def _layernorm(vec, eps=1e-6):
    n = len(vec)
    m = sum(vec) / n
    var = sum((v - m) ** 2 for v in vec) / n
    return [(v - m) / math.sqrt(var + eps) for v in vec]


def _linear(vec, w, b):
    out = []
    for r in range(len(b)):
        out.append(sum(v * w[r][c] for c, v in enumerate(vec)) + b[r])
    return out


def _mat(rows):
    return [list(r) for r in rows]


def _oracle_stream(x, temb, mod_w, mod_b, ff1_w, ff1_b, ff2_w, ff2_b, attn):
    t = [_silu(v) for v in temb]
    emb = _linear(t, mod_w, mod_b)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        emb[0:4], emb[4:8], emb[8:12], emb[12:16], emb[16:20], emb[20:24],
    )
    x_out = []
    for tok in x:
        n = _layernorm(tok)
        x_out.append([n[d] * (1 + scale_msa[d]) + shift_msa[d] for d in range(4)])
    # 上游残差加在「原输入」上（hidden_states = hidden_states + gate*attn），
    # 调制值只作为 attention 的输入
    attn_out = []
    for tok, a in zip(x, attn):
        attn_out.append([tok[d] + gate_msa[d] * a[d] for d in range(4)])
    ff_in = []
    for tok in attn_out:
        n = _layernorm(tok)
        ff_in.append([n[d] * (1 + scale_mlp[d]) + shift_mlp[d] for d in range(4)])
    ff_out = []
    for tok in ff_in:
        h = [_gelu_tanh(v) for v in _linear(tok, ff1_w, ff1_b)]
        ff_out.append(_linear(h, ff2_w, ff2_b))
    final = []
    for tok, f in zip(attn_out, ff_out):
        final.append([tok[d] + gate_mlp[d] * f[d] for d in range(4)])
    return final, emb, x_out


def _build_case():
    g = torch.Generator().manual_seed(649)
    D, Li, Lt = 4, 2, 3
    hidden = torch.randn(1, Li, D, generator=g).to(torch.float64)
    hidden[0, 0] = torch.tensor([1.0, 1.0 + 1e-3, 1.0 - 1e-3, 1.0 + 2e-3],
                                dtype=torch.float64)  # 近常数行，方差 ~1e-6，eps 主导
    enc = torch.randn(1, Lt, D, generator=g).to(torch.float64)
    temb = torch.randn(1, D, generator=g).to(torch.float64)
    attn_img = torch.randn(1, Li, D, generator=g).to(torch.float64)
    attn_txt = torch.randn(1, Lt, D, generator=g).to(torch.float64)
    w = {}
    for name, out_dim in (("img_mod", 24), ("txt_mod", 24), ("img_ff1", 16), ("txt_ff1", 16)):
        w[f"{name}_w"] = torch.randn(out_dim, D, generator=g).to(torch.float64) * 0.5
        w[f"{name}_b"] = torch.randn(out_dim, generator=g).to(torch.float64) * 0.1
    for name, out_dim in (("img_ff2", 4), ("txt_ff2", 4)):
        w[f"{name}_w"] = torch.randn(out_dim, 16, generator=g).to(torch.float64) * 0.5
        w[f"{name}_b"] = torch.randn(out_dim, generator=g).to(torch.float64) * 0.1
    return hidden, enc, temb, w, attn_img, attn_txt


def _py(v):
    return v.detach().to(torch.float64).tolist()


def _close(a, b, tol=1e-5):
    if isinstance(a, torch.Tensor):
        a = _py(a)
    flat_a, flat_b = [], []
    def rec(x, y):
        if isinstance(x, list):
            for u, v in zip(x, y):
                rec(u, v)
        else:
            flat_a.append(x)
            flat_b.append(y)
    rec(a, b)
    return all(abs(p - q) <= tol * max(1.0, abs(q)) for p, q in zip(flat_a, flat_b))


def check_oracle_agreement(solve):
    hidden, enc, temb, w, attn_img, attn_txt = _build_case()
    led = solve(hidden, enc, temb, w, attn_img, attn_txt)
    img_exp, img_emb, img_xout = _oracle_stream(_py(hidden[0]), _py(temb[0]),
                                      _py(w["img_mod_w"]), _py(w["img_mod_b"]),
                                      _py(w["img_ff1_w"]), _py(w["img_ff1_b"]),
                                      _py(w["img_ff2_w"]), _py(w["img_ff2_b"]), _py(attn_img[0]))
    txt_exp, txt_emb, txt_xout = _oracle_stream(_py(enc[0]), _py(temb[0]),
                                      _py(w["txt_mod_w"]), _py(w["txt_mod_b"]),
                                      _py(w["txt_ff1_w"]), _py(w["txt_ff1_b"]),
                                      _py(w["txt_ff2_w"]), _py(w["txt_ff2_b"]), _py(attn_txt[0]))
    assert _close(_py(led["hidden_states"][0]), img_exp)
    assert _close(_py(led["encoder_hidden_states"][0]), txt_exp)
    assert _close(_py(led["img_mod"].flatten()), img_emb)
    assert _close(_py(led["txt_mod"].flatten()), txt_emb)
    # 调制后的 attention 输入：oracle 的 x_out 即该值
    assert _close(_py(led["img_norm_input"]), img_xout)
    assert _close(_py(led["txt_norm_input"]), txt_xout)


def check_mod_order(solve):
    # img_mod 的六行必须按 chunk 顺序 = linear 输出的六个 D 段
    hidden, enc, temb, w, ai, at = _build_case()
    led = solve(hidden, enc, temb, w, ai, at)
    emb = torch.nn.functional.linear(
        torch.nn.functional.silu(temb), w["img_mod_w"], w["img_mod_b"])[0]
    assert led["img_mod"].shape == (6, 4)
    for k in range(6):
        assert torch.allclose(led["img_mod"][k], emb[k * 4:(k + 1) * 4], atol=1e-6)
    emb_t = torch.nn.functional.linear(
        torch.nn.functional.silu(temb), w["txt_mod_w"], w["txt_mod_b"])[0]
    for k in range(6):
        assert torch.allclose(led["txt_mod"][k], emb_t[k * 4:(k + 1) * 4], atol=1e-6)


def check_two_streams_share_temb(solve):
    # 两路共用 temb 但调制权重独立：img_mod 与 txt_mod 不同
    hidden, enc, temb, w, ai, at = _build_case()
    led = solve(hidden, enc, temb, w, ai, at)
    assert not torch.allclose(led["img_mod"], led["txt_mod"], atol=1e-6)


def check_zero_gate_identity(solve):
    # gate_msa = gate_mlp = 0 时输出 = 输入（残差恒等）
    hidden, enc, temb, w, ai, at = _build_case()
    D = 4
    for name in ("img_mod_w", "txt_mod_w"):
        w = dict(w)
        wm = w[name].clone()
        wm[8:12] = 0.0   # gate_msa 行
        wm[20:24] = 0.0  # gate_mlp 行
        w[name] = wm
    for name in ("img_mod_b", "txt_mod_b"):
        w = dict(w)
        b = w[name].clone()
        b[8:12] = 0.0   # gate_msa
        b[20:24] = 0.0  # gate_mlp
        w[name] = b
    led = solve(hidden, enc, temb, w, ai, at)
    assert torch.allclose(led["hidden_states"], hidden, atol=1e-5)
    assert torch.allclose(led["encoder_hidden_states"], enc, atol=1e-5)


def check_scale_shift_effect(solve):
    # scale_mlp/shift_mlp 作用于 ff 分支输入：改权重必改变输出方向。
    # （scale_msa/shift_msa 只影响 attention 的输入，而本题 attention 输出是
    #   显式入参，对块输出无影响——这也解释了 gate 归零即残差恒等。）
    hidden, enc, temb, w, ai, at = _build_case()
    led1 = solve(hidden, enc, temb, w, ai, at)
    w2 = dict(w)
    w2["img_mod_w"] = w["img_mod_w"].clone()
    w2["img_mod_w"][16:20] += 1.0  # scale_mlp 行
    led2 = solve(hidden, enc, temb, w2, ai, at)
    assert not torch.allclose(led1["hidden_states"], led2["hidden_states"], atol=1e-6)
    # 文本流不受图像流权重影响
    assert torch.allclose(led1["encoder_hidden_states"], led2["encoder_hidden_states"], atol=1e-9)


def check_noncontiguous(solve):
    hidden, enc, temb, w, ai, at = _build_case()
    big = torch.randn(1, 6, 4, generator=torch.Generator().manual_seed(1)).to(torch.float64)
    nc = big[:, ::3, :][:, :2, :]
    assert not nc.is_contiguous()
    led = solve(nc, enc, temb, w, ai, at)
    assert led["hidden_states"].shape == (1, 2, 4)


def check_invalid_batch(solve):
    hidden, enc, temb, w, ai, at = _build_case()
    try:
        solve(torch.cat([hidden, hidden], 0), enc, temb, w, ai, at)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for batch != 1")
    try:
        solve("bad", enc, temb, w, ai, at)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-tensor")
