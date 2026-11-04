"""Zipformer2 流式状态账——判题用例（order 828）。

约定：torch.set_default_dtype(float64)；每个 check 独立 manual_seed；
无占位锚值（全部断言为闭式对拍/恒等式，无 RNG 漂移风险）。
"""

import math

import torch
from torch.nn import functional as F


def check(solve):
    check_cache_roll(solve)
    check_attn_scores(solve)
    check_value_out(solve)
    check_nonlin_out(solve)
    check_conv_gated(solve)
    check_chunk_scale(solve)
    check_chunk_causal_conv(solve)
    print("all checks passed")


def check_cache_roll(solve):
    """滚动缓存：拼接保序、截断取尾、长度恒 L、双帧滚动、0 长度、dim=2。"""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    L, seq, b, d = 3, 4, 2, 5
    cache = torch.randn(L, b, d)
    x = torch.randn(seq, b, d)
    out = solve(mode="cache_roll", x=x, cache=cache, dim=0, left_context_len=L)
    assert out["padded"].shape == (L + seq, b, d)
    assert torch.equal(out["padded"], torch.cat([cache, x], dim=0))
    assert torch.equal(out["cache"], x[-L:])
    # 双帧滚动后窗口 = 最近 L 帧（cat 两段再截）
    x2 = torch.randn(2, b, d)
    out2 = solve(mode="cache_roll", x=x2, cache=out["cache"], dim=0,
                 left_context_len=L)
    assert torch.equal(out2["cache"], torch.cat([x, x2], dim=0)[-L:])
    # L=0 空缓存
    out3 = solve(mode="cache_roll", x=x, cache=torch.zeros(0, b, d), dim=0,
                 left_context_len=0)
    assert out3["cache"].shape == (0, b, d)
    # conv 帧轴（dim=2）同账
    xc = torch.randn(b, d, seq)
    cc = torch.randn(b, d, L)
    out4 = solve(mode="cache_roll", x=xc, cache=cc, dim=2, left_context_len=L)
    assert torch.equal(out4["cache"], xc[..., -L:])


def check_attn_scores(solve):
    """注意力打分：缓存滚动、q@k、相对位置逐项手算、mask、softmax 行。"""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1)
    L, seq, b, h, dq, dp = 2, 3, 1, 4, 3, 2
    q = torch.randn(seq, b, h * dq)
    k_new = torch.randn(seq, b, h * dq)
    cached_key = torch.randn(L, b, h * dq)
    p = torch.randn(seq, b, h * dp)
    pos_emb = torch.randn(1, 2 * seq - 1 + L, h * dp)
    pos_proj = torch.randn(h * dp, h * dp) * 0.5
    out = solve(mode="attn_scores", q=q, k=k_new, cached_key=cached_key, p=p,
                pos_emb=pos_emb, pos_proj=pos_proj, num_heads=h,
                key_padding_mask=None, left_context_len=L)
    assert out["weights"].shape == (h, b, seq, L + seq)
    assert torch.equal(out["cache"], k_new[-L:])
    k_all = torch.cat([cached_key, k_new], dim=0)
    qh = q.reshape(seq, b, h, dq).permute(2, 1, 0, 3)
    kh = k_all.reshape(L + seq, b, h, dq).permute(2, 1, 3, 0)
    scores = qh @ kh  # (h, b, seq, L+seq)
    pos = pos_emb @ pos_proj.t()  # (1, 2seq-1+L, h*dp)
    ph = p.reshape(seq, b, h, dp).permute(2, 1, 0, 3)
    # 逐项手算：pos_scores[h,:,t,j] = p[h,:,t,:] · pos[0, (seq-1-t)+j, :]
    # 注意 (h,b,dp)*(h,dp) 会在 b=1 时被广播成 (h,h,dp)——必须先对齐成 (h,1,dp)
    manual = torch.stack([
        torch.stack([(ph[:, :, t, :] *
                      pos[0, (seq - 1 - t) + j, :].reshape(h, 1, dp)).sum(-1)
                     for j in range(L + seq)], dim=-1) for t in range(seq)],
        dim=2)  # (h, b, seq, L+seq)
    assert torch.allclose(out["weights"],
                          (scores + manual).softmax(-1), atol=1e-12)
    # 相对索引语义钉死：rel = (seq-1-t)+j 且 rel 必须落在 pos_emb 域内
    for t in range(seq):
        assert 0 <= (seq - 1 - t) + (L + seq - 1) < 2 * seq - 1 + L
    # mask：-1000 填充 → 该列权重归零、行和仍为 1
    mask = torch.zeros(b, L + seq, dtype=torch.bool)
    mask[:, 0] = True
    out_m = solve(mode="attn_scores", q=q, k=k_new, cached_key=cached_key, p=p,
                  pos_emb=pos_emb, pos_proj=pos_proj, num_heads=h,
                  key_padding_mask=mask, left_context_len=L)
    assert (out_m["weights"][..., 0] < 1e-12).all()
    assert torch.allclose(out_m["weights"].sum(-1), torch.ones(h, b, seq),
                          atol=1e-12)
    # pos_proj 必须真参与（否则数值显著变化）
    out_np = solve(mode="attn_scores", q=q, k=k_new, cached_key=cached_key,
                   p=p, pos_emb=pos_emb,
                   pos_proj=torch.eye(h * dp), num_heads=h,
                   key_padding_mask=None, left_context_len=L)
    assert not torch.allclose(out["weights"], out_np["weights"], atol=1e-6)


def check_value_out(solve):
    """值聚合：weights @ [cache; v] 闭式对拍、缓存滚动、逐帧 einsum 对拍。"""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(2)
    L, seq, b, h, dv = 2, 3, 1, 4, 3
    x = torch.randn(seq, b, h * dv)
    cached = torch.randn(L, b, h * dv)
    w = torch.randn(h, b, seq, L + seq).softmax(-1)
    out = solve(mode="value_out", x=x, cached_val=cached, weights=w,
                left_context_len=L)
    assert out["out"].shape == x.shape
    assert torch.equal(out["cache"], x[-L:])
    xh = torch.cat([cached, x], 0).reshape(L + seq, b, h, dv).permute(2, 1, 0, 3)
    expect = (w @ xh).permute(2, 1, 0, 3).reshape(seq, b, h * dv)
    assert torch.allclose(out["out"], expect, atol=1e-12)
    # 逐帧手算
    for t in range(seq):
        manual = torch.einsum("hbj,hbjd->hbd", w[:, :, t, :], xh)
        assert torch.allclose(out["out"][t].reshape(b, h, dv).permute(1, 0, 2),
                              manual, atol=1e-12), t


def check_nonlin_out(solve):
    """nonlin_attention：s/x/y 三分、tanh 门控、缓存 cat、weights 聚合。"""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(3)
    L, seq, b, h, dd = 2, 3, 1, 4, 3
    x = torch.randn(seq, b, 3 * h * dd)
    cached = torch.randn(h, b, L, dd)
    w = torch.randn(h, b, seq, L + seq).softmax(-1)
    out = solve(mode="nonlin_out", x=x, attn_weights=w, cached_x=cached,
                left_context_len=L)
    assert out["out"].shape == (seq, b, h * dd)
    assert out["cache"].shape == (h, b, L, dd)
    xr = x.reshape(seq, b, 3, h, dd)
    s = torch.tanh(xr[:, :, 0])
    xv = xr[:, :, 1] * s
    xh = xv.permute(2, 1, 0, 3)  # (h, b, seq, dd)
    xpad = torch.cat([cached, xh], dim=2)
    assert torch.equal(out["cache"], xh[:, :, -L:, :])
    expect = (w @ xpad).permute(2, 1, 0, 3).reshape(seq, b, h * dd)
    assert torch.allclose(out["out"], expect, atol=1e-12)
    # tanh 门必须真实生效
    assert not torch.allclose(xh, xr[:, :, 1].permute(1, 2, 0, 3), atol=1e-3)
    # 缓存窗口深度 L 正确（改 cached 的最旧帧应影响输出，最新帧才不进窗口外）
    cached_mut = cached.clone()
    cached_mut[:, :, 0, :] += 5.0
    out_mut = solve(mode="nonlin_out", x=x, attn_weights=w, cached_x=cached_mut,
                    left_context_len=L)
    assert not torch.allclose(out["out"], out_mut["out"], atol=1e-9)


def check_conv_gated(solve):
    """卷积门控：in_proj 双倍通道、sigmoid 门、mask 置零。"""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(4)
    t, b, ch = 5, 2, 6
    x = torch.randn(t, b, ch)
    w = torch.randn(2 * ch, ch) * 0.3
    bb = torch.randn(2 * ch)
    out = solve(mode="conv_gated", x=x, in_proj_w=w, in_proj_b=bb, mask=None)
    assert out["gated"].shape == (b, ch, t)
    proj = x @ w.t() + bb
    a, g = proj[..., :ch], proj[..., ch:]
    expect = (a * torch.sigmoid(g)).permute(1, 2, 0)
    assert torch.allclose(out["gated"], expect, atol=1e-12)
    # mask 置零：被 mask 的 (batch, time) 位置全通道为 0，其余帧不变
    mask = torch.zeros(b, t, dtype=torch.bool)
    mask[0, 2] = True
    out_m = solve(mode="conv_gated", x=x, in_proj_w=w, in_proj_b=bb, mask=mask)
    assert torch.all(out_m["gated"][0, :, 2] == 0)
    assert torch.allclose(out_m["gated"][:, :, 0], expect[:, :, 0],
                          atol=1e-12)
    # 门必须真参与
    assert not torch.allclose(out["gated"], a.permute(1, 2, 0), atol=1e-3)


def check_chunk_scale(solve):
    """chunk_scale：1+头尾修正，长块补零、短块截断、k 块重叠相加。"""
    torch.set_default_dtype(torch.float64)
    ch, k = 3, 5
    scale = torch.randn(2, ch, k) * 0.5
    out = solve(mode="chunk_scale", scale=scale, chunk_size=10)
    assert out["scale"].shape == (ch, 10)
    # cs=10 > k=5：左修正贴列 0-4、右修正贴列 5-9，无重叠
    assert torch.allclose(out["scale"][:, :5], 1.0 + scale[0], atol=1e-12)
    assert torch.allclose(out["scale"][:, 5:], 1.0 + scale[1], atol=1e-12)
    # cs=8：左占列 0-4、右占列 3-7，重叠列 3-4 相加
    out8 = solve(mode="chunk_scale", scale=scale, chunk_size=8)
    assert torch.allclose(out8["scale"][:, :3],
                          1.0 + scale[0][:, :3], atol=1e-12)
    assert torch.allclose(out8["scale"][:, 3:5],
                          1.0 + scale[0][:, 3:5] + scale[1][:, :2],
                          atol=1e-12)
    assert torch.allclose(out8["scale"][:, 5:],
                          1.0 + scale[1][:, 2:5], atol=1e-12)
    # chunk_size == k：全行 = 1 + left + right
    out_k = solve(mode="chunk_scale", scale=scale, chunk_size=5)
    assert torch.allclose(out_k["scale"], 1.0 + scale[0] + scale[1],
                          atol=1e-12)
    # chunk_size < k：left 取前 chunk_size、right 取后 chunk_size
    out_s = solve(mode="chunk_scale", scale=scale, chunk_size=3)
    assert torch.allclose(out_s["scale"][:, :3],
                          1.0 + scale[0][:, :3] + scale[1][:, -3:],
                          atol=1e-12)
    # cs=9：左占列 0-4、右占列 4-8，仅列 4 重叠
    out9 = solve(mode="chunk_scale", scale=scale, chunk_size=9)
    assert torch.allclose(out9["scale"][:, 4],
                          1.0 + scale[0][:, 4] + scale[1][:, 0], atol=1e-12)
    assert torch.allclose(out9["scale"][:, 0], 1.0 + scale[0][:, 0],
                          atol=1e-12)
    assert torch.allclose(out9["scale"][:, 8], 1.0 + scale[1][:, 4],
                          atol=1e-12)


def check_chunk_causal_conv(solve):
    """chunk 级因果卷积：分解对拍、cache 滚动、整除恒等、双块走账、
    因果性（改未来帧不影响当前块）、chunk_size=1 边界。"""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(5)
    b, ch, seq, cs, k = 2, 3, 8, 4, 5
    half_k = (k + 1) // 2
    left_pad = k // 2
    x = torch.randn(b, ch, seq)
    cache = torch.randn(b, ch, left_pad)
    causal_w = torch.randn(ch, 1, half_k) * 0.4
    causal_b = torch.randn(ch) * 0.1
    chunk_w = torch.randn(ch, 1, k) * 0.4
    chunk_b = torch.randn(ch) * 0.1
    scale = torch.randn(2, ch, k) * 0.3
    out = solve(mode="chunk_causal_conv", x=x[..., :cs], cache=cache,
                causal_w=causal_w, causal_b=causal_b, chunk_w=chunk_w,
                chunk_b=chunk_b, scale=scale, chunk_size=cs)
    assert out["out"].shape == x[..., :cs].shape
    assert torch.equal(out["cache"],
                       torch.cat([cache, x[..., :cs]], -1)[..., -left_pad:])
    # 分解对拍（单块 seq==cs）
    xp = torch.cat([cache, x[..., :cs]], -1)
    x_causal = F.conv1d(xp, causal_w, causal_b, groups=ch)
    x_chunk = F.conv1d(xp[..., left_pad:], chunk_w, chunk_b, groups=ch,
                       padding=k // 2)
    cs_t = solve(mode="chunk_scale", scale=scale, chunk_size=cs)["scale"]
    assert torch.allclose(out["out"], x_chunk * cs_t.unsqueeze(0) + x_causal,
                          atol=1e-12)
    # 整除恒等：逐块流式拼接 == 一次性大块
    eq = solve(mode="stream_equiv", x=x, cache=cache, causal_w=causal_w,
               causal_b=causal_b, chunk_w=chunk_w, chunk_b=chunk_b,
               scale=scale, chunk_size=cs)
    assert torch.allclose(torch.cat(eq["outs"], dim=-1), eq["whole"],
                          atol=1e-12)
    assert torch.allclose(eq["caches"][-1],
                          torch.cat([cache, x], -1)[..., -left_pad:],
                          atol=1e-12)
    # 双块显式走账
    r1 = solve(mode="chunk_causal_conv", x=x[..., :cs], cache=cache,
               causal_w=causal_w, causal_b=causal_b, chunk_w=chunk_w,
               chunk_b=chunk_b, scale=scale, chunk_size=cs)
    r2 = solve(mode="chunk_causal_conv", x=x[..., cs:], cache=r1["cache"],
               causal_w=causal_w, causal_b=causal_b, chunk_w=chunk_w,
               chunk_b=chunk_b, scale=scale, chunk_size=cs)
    assert torch.allclose(torch.cat([r1["out"], r2["out"]], -1),
                          eq["whole"], atol=1e-12)
    # 纯因果性：当前块输出不依赖未来帧
    x_mut = x.clone()
    x_mut[..., cs:] += 1.0
    r1m = solve(mode="chunk_causal_conv", x=x_mut[..., :cs], cache=cache,
                causal_w=causal_w, causal_b=causal_b, chunk_w=chunk_w,
                chunk_b=chunk_b, scale=scale, chunk_size=cs)
    assert torch.allclose(r1["out"], r1m["out"], atol=1e-12)
    # causal 支路真因果：cache 末帧变化必影响块首输出
    cache_mut = cache.clone()
    cache_mut[..., -1] += 2.0
    r1c = solve(mode="chunk_causal_conv", x=x[..., :cs], cache=cache_mut,
                causal_w=causal_w, causal_b=causal_b, chunk_w=chunk_w,
                chunk_b=chunk_b, scale=scale, chunk_size=cs)
    assert not torch.allclose(r1["out"][..., 0], r1c["out"][..., 0],
                              atol=1e-9)
    # chunk_size=1 边界：scale 截断语义下的闭式对拍
    out1 = solve(mode="chunk_causal_conv", x=x[..., :1], cache=cache,
                 causal_w=causal_w, causal_b=causal_b, chunk_w=chunk_w,
                 chunk_b=chunk_b, scale=scale, chunk_size=1)
    s1 = solve(mode="chunk_scale", scale=scale, chunk_size=1)["scale"]
    x_c1 = F.conv1d(torch.cat([cache, x[..., :1]], -1), causal_w, causal_b,
                    groups=ch)
    x_g1 = (F.conv1d(torch.cat([cache, x[..., :1]], -1)[..., left_pad:],
                     chunk_w, chunk_b, groups=ch, padding=k // 2)
            * s1.unsqueeze(0))
    assert torch.allclose(out1["out"], x_g1 + x_c1, atol=1e-12)
