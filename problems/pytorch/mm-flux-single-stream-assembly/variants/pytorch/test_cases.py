"""独立 oracle + 锚点用例：mm-flux-single-stream-assembly (order 686).

oracle 与 solution 独立实现：标量循环手写 silu/gelu_tanh/layer_norm/
matmul，张量拼接用列表逐 token 组装，避免共享任何 torch 调用。
"""

import math

import torch
import torch.nn.functional as F


def _silu(x):
    return x / (1.0 + math.exp(-x)) if x > -60 else 0.0


def _gelu_tanh(x):
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _layer_norm_row(row, eps=1e-6):
    n = len(row)
    mu = sum(row) / n
    var = sum((v - mu) ** 2 for v in row) / n
    return [(v - mu) / math.sqrt(var + eps) for v in row]


def _linear_row(row, W, b):
    return [sum(row[i] * W[j][i] for i in range(len(row))) + b[j] for j in range(len(b))]


def _oracle(enc_out, img_out, t_emb, pooled_proj, weights, attn_output):
    B, Lt, D = enc_out.shape
    Li = img_out.shape[1]
    mod_w = weights["single_mod_w"].tolist()
    mod_b = weights["single_mod_b"].tolist()
    mlp_w = weights["proj_mlp_w"].tolist()
    mlp_b = weights["proj_mlp_b"].tolist()
    out_w = weights["proj_out_w"].tolist()
    out_b = weights["proj_out_b"].tolist()
    M = len(mlp_b)

    vec_rows, shift_rows, scale_rows, gate_rows, final = [], [], [], [], []
    for bi in range(B):
        vec = [t_emb[bi][j] + pooled_proj[bi][j] for j in range(D)]
        mod = _linear_row([_silu(v) for v in vec], mod_w, mod_b)
        shift, scale, gate = mod[:D], mod[D:2 * D], mod[2 * D:]

        # 拼流：文本在前
        rows = [enc_out[bi][t][:] for t in range(Lt)] + [img_out[bi][t][:] for t in range(Li)]
        residual = [r[:] for r in rows]
        norm_rows = [_layer_norm_row(r) for r in rows]
        norm_x = [
            [norm_rows[t][j] * (1 + scale[j]) + shift[j] for j in range(D)]
            for t in range(Lt + Li)
        ]
        mlp = [[_gelu_tanh(v) for v in _linear_row(norm_x[t], mlp_w, mlp_b)] for t in range(Lt + Li)]
        outs = []
        for t in range(Lt + Li):
            fused = list(attn_output[bi][t]) + mlp[t]
            o = _linear_row(fused, out_w, out_b)
            outs.append([gate[j] * o[j] for j in range(D)])
        final.append([[residual[t][j] + outs[t][j] for j in range(D)] for t in range(Lt, Lt + Li)])

        vec_rows.append(vec)
        shift_rows.append(shift)
        scale_rows.append(scale)
        gate_rows.append(gate)

    mk = lambda rows: torch.tensor(rows, dtype=enc_out.dtype)
    return (mk(vec_rows), mk(shift_rows), mk(scale_rows), mk(gate_rows),
            mk(final))


def _compare(solve, enc, img, te, pp, w, attn):
    exp = _oracle(enc, img, te, pp, w, attn)
    got = solve(enc, img, te, pp, w, attn)
    assert len(got) == 5, "must return a 5-tuple"
    for i, (e, g) in enumerate(zip(exp, got)):
        assert torch.allclose(e, g, atol=1e-5, rtol=1e-5), f"part {i}\nexp={e}\ngot={g}"
        assert e.dtype == g.dtype, f"part {i} dtype {g.dtype} != {e.dtype}"
    return got


def _mk(B, Lt, Li, D, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = lambda *s: torch.randn(*s, generator=g, dtype=dtype)
    enc = t(B, Lt, D)
    img = t(B, Li, D)
    te = t(B, D)
    pp = t(B, D)
    M = 3 * D // 2
    w = {
        "single_mod_w": t(3 * D, D) * 0.3,
        "single_mod_b": t(3 * D) * 0.3,
        "proj_mlp_w": t(M, D) * 0.3,
        "proj_mlp_b": t(M) * 0.3,
        "proj_out_w": t(D, D + M) * 0.3,
        "proj_out_b": t(D) * 0.3,
    }
    attn = t(B, Lt + Li, D) * 0.5
    return enc, img, te, pp, w, attn


def _approx(a, b, tol=1e-4):
    assert abs(a - b) < tol, f"{a} != {b}"


def check_basic_shape_and_vec(solve):
    enc, img, te, pp, w, attn = _mk(1, 2, 3, 8, seed=1)
    vec, sh, sc, ga, final = _compare(solve, enc, img, te, pp, w, attn)
    assert vec.shape == (1, 8) and sh.shape == (1, 8) and final.shape == (1, 3, 8)
    # vec 是加法：等于 t_emb + pooled_proj 逐元素
    manual = (te + pp)[0]
    for j in range(8):
        _approx(vec[0][j].item(), manual[j].item())


def check_vec_is_add_not_concat(solve):
    # 钉死加法：若实现是 cat(2D) 则形状/数值全错
    enc, img, te, pp, w, attn = _mk(2, 2, 2, 8, seed=2)
    vec, *_ , final = _compare(solve, enc, img, te, pp, w, attn)
    assert vec.shape == (2, 8)
    assert torch.equal(vec, te + pp)


def check_text_first_concat_order(solve):
    # 文本流在前：交换 enc/img（attn 行同步换）后结果必然不同，证明顺序在账上
    enc, img, te, pp, w, attn = _mk(1, 2, 2, 8, seed=3)
    led = _compare(solve, enc, img, te, pp, w, attn)
    led2 = _compare(solve, img, enc, te, pp, w,
                    torch.cat([attn[:, 2:], attn[:, :2]], dim=1))
    assert not torch.allclose(led[4], led2[4])


def check_three_chunks_not_six(solve):
    # single block 是 3 参数调制：shift/scale/gate 都来自同一个 [3D] 的三等分
    enc, img, te, pp, w, attn = _mk(1, 1, 1, 6, seed=4)
    vec, sh, sc, ga, final = _compare(solve, enc, img, te, pp, w, attn)
    mod = F.linear(F.silu(vec), w["single_mod_w"], w["single_mod_b"])
    assert torch.allclose(sh, mod[:, :6], atol=1e-6)
    assert torch.allclose(sc, mod[:, 6:12], atol=1e-6)
    assert torch.allclose(ga, mod[:, 12:], atol=1e-6)


def check_mlp_shares_normed_input(solve):
    # norm_x 同源喂 attn 与 mlp：把 attn_output 换成 norm_x 的拷贝，
    # 则 fused 的前半段与 mlp 输入同源（间接验证 mlp 用的是调制后的 norm_x）
    enc, img, te, pp, w, attn = _mk(1, 2, 2, 8, seed=5)
    vec, sh, sc, ga, final = _compare(solve, enc, img, te, pp, w, attn)
    hidden = torch.cat([enc, img], dim=1)
    nx = F.layer_norm(hidden, (8,), None, None, 1e-6) * (1 + sc)[:, None] + sh[:, None]
    # 用 nx 替代 attn_output 再跑：fused 前半 == nx，等于把 attn 换成归一化流
    led2 = _compare(solve, enc, img, te, pp, w, nx)
    # 手工复算 proj_out 输入的前半段必须等于 nx
    mlp = F.gelu(F.linear(nx, w["proj_mlp_w"], w["proj_mlp_b"]), approximate="tanh")
    fused = torch.cat([nx, mlp], dim=2)
    manual = torch.cat([enc, img], dim=1) + (ga[:, None] * F.linear(fused, w["proj_out_w"], w["proj_out_b"]))
    assert torch.allclose(led2[4], manual[:, 2:], atol=1e-5)


def check_scale_one_plus(solve):
    # (1+scale) 语义：shift/scale 两段权重与偏置全零 → 调制退化为纯 LayerNorm
    enc, img, te, pp, w, attn = _mk(1, 2, 2, 8, seed=6)
    w2 = {k: v.clone() for k, v in w.items()}
    D = 8
    for seg in (slice(0, D), slice(D, 2 * D)):   # shift、scale 段
        w2["single_mod_w"][seg] = 0.0
        w2["single_mod_b"][seg] = 0.0
    led = _compare(solve, enc, img, te, pp, w2, attn)
    hidden = torch.cat([enc, img], dim=1)
    nx = F.layer_norm(hidden, (8,), None, None, 1e-6)
    mlp = F.gelu(F.linear(nx, w2["proj_mlp_w"], w2["proj_mlp_b"]), approximate="tanh")
    fused = torch.cat([attn, mlp], dim=2)
    # gate 段未清零：用返回的 gate 账本（已与 oracle 对拍）补全乘法
    manual = hidden + led[3][:, None] * F.linear(fused, w2["proj_out_w"], w2["proj_out_b"])
    assert torch.allclose(led[4], manual[:, 2:], atol=1e-5)


def check_gate_residual(solve):
    # gate 段权重与偏置全零 → gate ≡ 0 → 输出 = 残差（图像段原样返回）
    enc, img, te, pp, w, attn = _mk(1, 2, 3, 8, seed=7)
    w2 = {k: v.clone() for k, v in w.items()}
    D = 8
    w2["single_mod_w"][2 * D:] = 0.0
    w2["single_mod_b"][2 * D:] = 0.0
    led = _compare(solve, enc, img, te, pp, w2, attn)
    assert torch.allclose(led[4][0], img[0], atol=1e-6)


def check_slice_back_by_text_len(solve):
    # 切回按 Lt：Lt 改变时 final 形状跟随 Li
    for Lt in (1, 3):
        enc, img, te, pp, w, attn = _mk(1, Lt, 2, 8, seed=8 + Lt)
        _, _, _, _, final = _compare(solve, enc, img, te, pp, w, attn)
        assert final.shape == (1, 2, 8)


def check_dtype_float64(solve):
    enc, img, te, pp, w, attn = _mk(1, 1, 2, 8, dtype=torch.float64, seed=11)
    vec, sh, sc, ga, final = _compare(solve, enc, img, te, pp, w, attn)
    assert vec.dtype == torch.float64 and final.dtype == torch.float64


def check_noncontiguous_input(solve):
    enc, img, te, pp, w, attn = _mk(1, 2, 2, 8, seed=12)
    big = torch.randn(1, 8, 8)
    big2 = torch.randn(1, 8, 8)
    enc2 = big.transpose(1, 2)[:, :2]   # 转置后切片 → 非连续
    img2 = big2.transpose(1, 2)[:, :2]
    assert not enc2.is_contiguous() and not img2.is_contiguous()
    led = _compare(solve, enc2, img2, te, pp, w, attn)
    led_ref = _compare(solve, enc2.contiguous(), img2.contiguous(), te, pp, w, attn)
    for a, b in zip(led, led_ref):
        assert torch.allclose(a, b, atol=1e-6)


def check_input_not_mutated(solve):
    enc, img, te, pp, w, attn = _mk(1, 2, 2, 8, seed=13)
    snapshots = [x.clone() for x in (enc, img, te, pp, attn)]
    _compare(solve, enc, img, te, pp, w, attn)
    for x, s in zip((enc, img, te, pp, attn), snapshots):
        assert torch.equal(x, s)


def check_invalid_inputs(solve):
    enc, img, te, pp, w, attn = _mk(1, 2, 2, 8, seed=14)

    def expect_err(*args):
        try:
            solve(*args)
        except ValueError:
            return
        raise AssertionError("expected ValueError")

    expect_err(enc.half(), img, te, pp, w, attn)          # dtype
    expect_err(enc, img, te.half(), pp, w, attn)          # dtype 不一致
    bad = te.clone(); bad[0, 0] = float("nan")
    expect_err(enc, img, bad, pp, w, attn)                # 非有限
    expect_err(enc.squeeze(0), img, te, pp, w, attn)      # 2-D enc
    expect_err(enc, img, te.squeeze(0), pp, w, attn)      # 1-D t_emb
    expect_err(enc, img, te[:, :4], pp, w, attn)          # t_emb 形状
    expect_err(enc, img, te, pp, w, attn[:, :3])          # attn 形状
    w2 = dict(w); w2.pop("proj_out_b")
    expect_err(enc, img, te, pp, w2, attn)                # 缺键
    w3 = dict(w); w3["single_mod_w"] = w3["single_mod_w"][:, :4]
    expect_err(enc, img, te, pp, w3, attn)                # 权重形状
    w4 = dict(w); w4["proj_mlp_w"] = w4["proj_mlp_w"].half()
    expect_err(enc, img, te, pp, w4, attn)                # 权重 dtype
