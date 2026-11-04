"""Zipformer2 流式状态账（回源：k2-fsa/icefall egs/librispeech/ASR/zipformer，
``zipformer.py`` 的 streaming_forward 族 + ``scaling.py`` 的
``ChunkCausalDepthwiseConv1d``）。官方把 encoder 端流式拆成四类缓存账：

1. **滚动缓存**（attn 的 key/val、nonlin_attn 的 x）：`x_pad = cat([cache, x], dim)`
   之后 `cache' = x_pad 的最后 left_context_len 帧`。窗口是"左上下文"而非
   KV cache——旧帧从右侧进入、从左侧滚出，长度恒为 L。
2. **相对位置对齐**：pos_scores 最后一轴是相对位置，用 as_strided 折叠为
   绝对 key 索引；等价的显式 gather 是 ``indexes[t, j] = (seq_len-1-t) + j``
   （rows = time1-1..0 递减，cols = 0..k_len-1）。
3. **门控卷积**：ConvolutionModule 的 in_proj 出双倍通道，后半走 sigmoid 作门
   `x = x * sigmoid(s)`；padding 帧在进 depthwise conv 前置零。
4. **chunk 级因果卷积**：ChunkCausalDepthwiseConv1d 把一个 k 阶 depthwise 卷积
   拆成 (a) half-kernel（(k+1)//2 阶）真因果卷积 `x_causal`（左侧补
   left_pad = k//2，流式时该 pad 来自 cache）；(b) 对称 k 阶 chunkwise 卷积
   `x_chunk`（无 padding，只在块内看邻居），乘 chunk_scale 后与 x_causal 相加。
   chunk_scale = 1 + left_edge(头部修正,长度 k,块内右侧补零到 chunk_size)
                  + right_edge(尾部修正,长度 k,块内左侧补零)，
   chunk_size < k 时改为截断：left 取前 chunk_size、right 取后 chunk_size。
   **整除恒等**：当 seq_len 是 chunk_size 整数倍时，逐块流式（cache 传左 pad）
   与一次性 forward(x, chunk_size) 的结果逐步相等。

所有权重/输入均为显式入参，纯张量重建，无 nn.Module。

判题约定：单入口 solve(mode, **kw)；CHECKS 列出全部校验函数；
check(solve) 逐个运行，任何 AssertionError 即失败。
"""

import math

import torch
from torch import Tensor
from torch.nn import functional as F

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _roll_cache(x: Tensor, cache: Tensor, dim: int, left_context_len: int):
    """官方滚动缓存：拼接后取最后 L 帧。（zipformer.py streaming_forward 族）"""
    padded = torch.cat([cache, x], dim=dim)
    if left_context_len == 0:
        new_cache = padded.narrow(dim, padded.shape[dim], 0)
    else:
        new_cache = padded.narrow(dim, padded.shape[dim] - left_context_len,
                                  left_context_len)
    return padded, new_cache.contiguous()


def _chunk_scale(scale: Tensor, chunk_size: int) -> Tensor:
    """官方 _get_chunk_scale：返回 (channels, chunk_size) 的 1+修正因子。"""
    left_edge = scale[0]
    right_edge = scale[1]
    k = scale.shape[-1]
    channels = scale.shape[1]
    if chunk_size < k:
        left_edge = left_edge[:, :chunk_size]
        right_edge = right_edge[:, -chunk_size:]
    else:
        pad = torch.zeros(channels, chunk_size - k, dtype=scale.dtype,
                          device=scale.device)
        left_edge = torch.cat((left_edge, pad), dim=-1)
        right_edge = torch.cat((pad, right_edge), dim=-1)
    return 1.0 + (left_edge + right_edge)


def solve(mode: str, **kw):
    """Zipformer2 流式账。各 mode 的参数与返回值见对应分支。"""
    if mode == "cache_roll":
        """通用滚动缓存。kw: x, cache, dim, left_context_len
        返回 {"padded", "cache"}（cache 恒为 left_context_len 帧，0 允许）。"""
        padded, new_cache = _roll_cache(kw["x"], kw["cache"], kw["dim"],
                                        kw["left_context_len"])
        return {"padded": padded, "cache": new_cache}

    if mode == "attn_scores":
        """注意力打分 + 相对位置对齐 + 缓存滚动。kw: q, p, cached_key, pos_proj,
        pos_emb, key_padding_mask（可为 None）, left_context_len
        q/p: (seq, b, h*dq)/(seq, b, h*dp)；cached_key: (L, b, h*dq)；
        pos_proj: (h*dp, h*dp)；pos_emb: (1|b, 2*seq-1+L, h*dp)。
        返回 {"weights", "cache"}：softmax 后的 (h, b, seq, L+seq) 与新 key 缓存。
        官方 streaming_forward（RelPositionMultiheadAttentionWeights）。"""
        q_in, p_in = kw["q"], kw["p"]
        L = kw["left_context_len"]
        seq, b, qd = q_in.shape
        num_heads = kw["num_heads"]
        dq = qd // num_heads
        pd = p_in.shape[-1] // num_heads

        padded, cached_key = _roll_cache(kw["k"], kw["cached_key"], 0, L)
        k_all = padded
        k_len = k_all.shape[0]

        qh = q_in.reshape(seq, b, num_heads, dq).permute(2, 1, 0, 3)
        ph = p_in.reshape(seq, b, num_heads, pd).permute(2, 1, 0, 3)
        kh = k_all.reshape(k_len, b, num_heads, dq).permute(2, 1, 3, 0)

        attn_scores = torch.matmul(qh, kh)

        pos = torch.matmul(kw["pos_emb"], kw["pos_proj"].t())
        seq_len2 = 2 * seq - 1 + L
        pos = pos.reshape(-1, seq_len2, num_heads, pd).permute(2, 0, 3, 1)
        pos_scores = torch.matmul(ph, pos)

        # as_strided 的显式 gather 等价：indexes[t, j] = (seq-1-t) + j
        rows = torch.arange(seq - 1, -1, -1, dtype=torch.long)
        cols = torch.arange(k_len, dtype=torch.long)
        indexes = rows.view(-1, 1) + cols.view(1, -1)
        n = pos_scores.shape[-1]
        pos_scores = pos_scores.reshape(num_heads * b, seq, n)
        pos_scores = pos_scores.gather(
            2, indexes.expand(num_heads * b, seq, k_len))
        pos_scores = pos_scores.reshape(num_heads, b, seq, k_len)

        attn_scores = attn_scores + pos_scores
        mask = kw.get("key_padding_mask")
        if mask is not None:
            attn_scores = attn_scores.masked_fill(
                mask.to(torch.bool).unsqueeze(1), -1000.0)
        weights = attn_scores.softmax(dim=-1)
        return {"weights": weights, "cache": cached_key}

    if mode == "value_out":
        """注意力值聚合 + 缓存滚动。kw: x, cached_val, weights, left_context_len
        x: (seq, b, h*dv)；cached_val: (L, b, h*dv)；
        weights: (h, b, seq, L+seq)。返回 {"out", "cache"}。
        官方 streaming_forward（SelfAttention）。"""
        L = kw["left_context_len"]
        seq, b, vd = kw["x"].shape
        h = kw["weights"].shape[0]
        dv = vd // h
        padded, cached_val = _roll_cache(kw["x"], kw["cached_val"], 0, L)
        xh = padded.reshape(L + seq, b, h, dv).permute(2, 1, 0, 3)
        out = torch.matmul(kw["weights"], xh)
        out = out.permute(2, 1, 0, 3).reshape(seq, b, vd)
        return {"out": out, "cache": cached_val}

    if mode == "nonlin_out":
        """nonlin_attention 流式账。kw: x, attn_weights, cached_x, left_context_len
        x: (seq, b, 3*h*dd)（s/x/y 三分）；cached_x: (h, b, L, dd)。
        返回 {"out", "cache"}。官方 streaming_forward（NonlinAttention）。"""
        L = kw["left_context_len"]
        seq, b, c3 = kw["x"].shape
        h = kw["attn_weights"].shape[0]
        dd = c3 // (3 * h)
        x = kw["x"].reshape(seq, b, 3, h, dd).permute(2, 0, 1, 3, 4)
        s, xv, _y = x[0], x[1], x[2]
        s = torch.tanh(s)
        xv = xv * s.unsqueeze(-1).reshape(seq, b, h, dd)
        xh = xv.permute(2, 1, 0, 3)  # (h, b, seq, dd)
        padded = torch.cat([kw["cached_x"], xh], dim=2)
        cached_x = padded[:, :, -L:, :] if L > 0 else padded[:, :, :0, :]
        out = torch.matmul(kw["attn_weights"], padded)
        out = out.permute(2, 1, 0, 3).reshape(seq, b, h * dd)
        return {"out": out, "cache": cached_x.contiguous()}

    if mode == "conv_gated":
        """卷积门控 + padding 置零。kw: x, in_proj_w, in_proj_b, mask（可选）
        x: (time, b, ch)；in_proj_w: (2*ch, ch)、in_proj_b: (2*ch)。
        返回 {"gated"}：(b, ch, time)，padding 帧已置零。
        官方 ConvolutionModule.streaming_forward 的门控段。"""
        x = torch.einsum("oc,tbc->tbo", kw["in_proj_w"], kw["x"]) + kw["in_proj_b"]
        x, s = x.chunk(2, dim=2)
        s = torch.sigmoid(s)
        x = x * s
        x = x.permute(1, 2, 0)  # (b, ch, time)
        mask = kw.get("mask")
        if mask is not None:
            x = x.masked_fill(
                mask.to(torch.bool).unsqueeze(1).expand_as(x), 0.0)
        return {"gated": x}

    if mode == "chunk_causal_conv":
        """chunk 级因果 depthwise 卷积（单块，含缓存进出）。
        kw: x (b, ch, seq), cache (b, ch, left_pad), causal_w (ch,1,half_k),
        causal_b (ch), chunk_w (ch,1,k), chunk_b (ch|None), scale (2, ch, k),
        chunk_size（缺省 seq）
        返回 {"out", "cache"}。官方 streaming_forward
        （ChunkCausalDepthwiseConv1d）。"""
        x, cache = kw["x"], kw["cache"]
        k = kw["chunk_w"].shape[-1]
        left_pad = k // 2
        seq = x.shape[-1]
        chunk_size = kw.get("chunk_size", seq)
        # 官方 streaming_forward 是单 chunk 语义：每次调用进一个 chunk
        assert seq == chunk_size, (seq, chunk_size)
        padded, new_cache = _roll_cache(x, cache, 2, left_pad)
        causal_w = kw["causal_w"]
        half_k = causal_w.shape[-1]
        x_causal = F.conv1d(padded, causal_w, kw["causal_b"], groups=x.shape[1])
        assert x_causal.shape == x.shape
        x_chunk = padded[..., left_pad:]
        # 对称 chunkwise conv 自带 padding=k//2（构造时即如此），形状不变
        x_chunk = F.conv1d(x_chunk, kw["chunk_w"],
                           kw.get("chunk_b"), groups=x.shape[1],
                           padding=k // 2)
        cs = _chunk_scale(kw["scale"], chunk_size)
        x_chunk = x_chunk * cs.unsqueeze(0)
        return {"out": x_chunk + x_causal, "cache": new_cache}

    if mode == "chunk_scale":
        """chunk_scale 因子表。kw: scale (2, ch, k), chunk_size
        返回 {"scale"}：(ch, chunk_size)。官方 _get_chunk_scale。"""
        return {"scale": _chunk_scale(kw["scale"], kw["chunk_size"])}

    if mode == "stream_equiv":
        """整除恒等探针：把一次大块卷积拆成若干次流式块卷积。
        kw: 同 chunk_causal_conv，另需 chunk_size 且 seq % chunk_size == 0。
        返回 {"outs", "caches", "whole"}——whole 为官方 forward(x, chunk_size)
        语义（全长 causal conv + 按 chunk_size 分块的 chunkwise conv）。
        恒等：逐块 out 拼接 == whole；末块 cache == whole 的 cache。"""
        x, cache = kw["x"], kw["cache"]
        b, ch, seq = x.shape
        cs = kw["chunk_size"]
        k = kw["chunk_w"].shape[-1]
        left_pad = k // 2
        assert cs > 0 and seq % cs == 0
        outs, caches = [], []
        cur = cache
        for t in range(0, seq, cs):
            r = solve("chunk_causal_conv", x=x[..., t:t + cs], cache=cur,
                      causal_w=kw["causal_w"], causal_b=kw["causal_b"],
                      chunk_w=kw["chunk_w"], chunk_b=kw.get("chunk_b"),
                      scale=kw["scale"], chunk_size=cs)
            outs.append(r["out"])
            caches.append(r["cache"])
            cur = r["cache"]
        # 官方 forward 语义：cache 相当于手工 left_pad；right_pad 补齐分块
        right_pad = -seq % cs
        xp = torch.cat([cache, x], dim=-1)
        x_causal = F.conv1d(xp, kw["causal_w"], kw["causal_b"], groups=ch)
        assert x_causal.shape[-1] == seq
        x_chunk_in = xp[..., left_pad:]
        if right_pad:
            x_chunk_in = F.pad(x_chunk_in, (0, right_pad))
        num_chunks = x_chunk_in.shape[-1] // cs
        xc = x_chunk_in.reshape(b, ch, num_chunks, cs)
        xc = xc.permute(0, 2, 1, 3).reshape(b * num_chunks, ch, cs)
        xc = F.conv1d(xc, kw["chunk_w"], kw.get("chunk_b"), groups=ch,
                      padding=k // 2)
        xc = xc * _chunk_scale(kw["scale"], cs).unsqueeze(0)
        xc = xc.reshape(b, num_chunks, ch, cs).permute(0, 2, 1, 3)
        xc = xc.reshape(b, ch, num_chunks * cs)[..., :seq]
        whole = xc + x_causal
        return {"outs": outs, "caches": caches, "whole": whole}

    raise ValueError(f"unknown mode: {mode}")


@check
def check_cache_roll(solve):
    """滚动缓存：拼接保序、截断取尾、长度恒 L；0 长度缓存允许。"""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    L, seq, b, d = 3, 4, 2, 5
    cache = torch.randn(L, b, d)
    x = torch.randn(seq, b, d)
    out = solve(mode="cache_roll", x=x, cache=cache, dim=0, left_context_len=L)
    assert out["padded"].shape == (L + seq, b, d)
    assert torch.equal(out["padded"], torch.cat([cache, x], dim=0))
    assert torch.equal(out["cache"], x[-L:])
    # 滚两帧：窗口内容等于最近 L 帧
    x2 = torch.randn(2, b, d)
    out2 = solve(mode="cache_roll", x=x2, cache=out["cache"], dim=0,
                 left_context_len=L)
    expect = torch.cat([x, x2], dim=0)[-L:]
    assert torch.equal(out2["cache"], expect)
    # L=0：空缓存
    out3 = solve(mode="cache_roll", x=x, cache=torch.zeros(0, b, d), dim=0,
                 left_context_len=0)
    assert out3["cache"].shape == (0, b, d)
    # dim=2（conv 的帧轴）同账
    xc = torch.randn(b, d, seq)
    cc = torch.randn(b, d, L)
    out4 = solve(mode="cache_roll", x=xc, cache=cc, dim=2, left_context_len=L)
    assert torch.equal(out4["cache"], xc[..., -L:])


@check
def check_attn_scores(solve):
    """注意力打分：q@k + 相对位置 gather 对拍、mask、softmax、缓存滚动。"""
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
    # 缓存滚动
    assert torch.equal(out["cache"], k_new[-L:])
    # q@k 对拍（用拼接后的 key）
    k_all = torch.cat([cached_key, k_new], dim=0)
    qh = q.reshape(seq, b, h, dq).permute(2, 1, 0, 3)
    kh = k_all.reshape(L + seq, b, h, dq).permute(2, 1, 3, 0)
    scores = qh @ kh
    # 相对位置 gather 对拍：绝对索引 (seq-1-t)+j
    pos = pos_emb @ pos_proj.t()  # (1, 2seq-1+L, h*dp)
    ph = p.reshape(seq, b, h, dp).permute(2, 1, 0, 3)
    for t in range(seq):
        for j in range(L + seq):
            rel = (seq - 1 - t) + j  # 官方 rows 递减 + cols
            contrib = (ph[:, :, t, :] * pos[0, rel, :].reshape(h, dp)).sum(-1)
            assert torch.allclose(
                scores[:, :, t, j] + contrib, out["weights"].reshape(
                    h, b, seq, L + seq)[:, :, t, j] * 0
                + (qh @ kh)[:, :, t, j] + contrib, atol=1e-12), (t, j)
    # 手工核对 softmax 行
    manual = (scores + torch.stack([
        torch.stack([(ph[:, :, t, :] *
                      pos[0, (seq - 1 - t) + j, :].reshape(h, dp)).sum(-1)
                     for j in range(L + seq)], dim=-1) for t in range(seq)],
        dim=-2))
    assert torch.allclose(out["weights"], manual.softmax(-1), atol=1e-12)
    # mask：置 -1000 后该列权重 ≈ 0
    mask = torch.zeros(b, L + seq, dtype=torch.bool)
    mask[:, 0] = True
    out_m = solve(mode="attn_scores", q=q, k=k_new, cached_key=cached_key, p=p,
                  pos_emb=pos_emb, pos_proj=pos_proj, num_heads=h,
                  key_padding_mask=mask, left_context_len=L)
    assert (out_m["weights"][..., 0] < 1e-12).all()
    assert torch.allclose(out_m["weights"].sum(-1),
                          torch.ones(h, b, seq), atol=1e-12)


@check
def check_value_out(solve):
    """值聚合：weights @ [cache; v] 与逐帧窗口重算等价；缓存滚动。"""
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
    # 行和为 1 的权重下，输出与手写循环一致
    for t in range(seq):
        manual = (w[:, :, t, :] @ xh[:, :, :, :].permute(0, 1, 3, 2)
                  if False else torch.einsum("hbj,hbjd->hbd", w[:, :, t, :],
                                             xh))
        assert torch.allclose(out["out"][t].reshape(b, h, dv).permute(1, 0, 2),
                              manual, atol=1e-12), t


@check
def check_nonlin_out(solve):
    """nonlin_attention：三分门控 + 缓存 cat + weights 聚合。"""
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
    # 门控手算
    xr = x.reshape(seq, b, 3, h, dd)
    s = torch.tanh(xr[:, :, 0])
    xv = xr[:, :, 1] * s
    xh = xv.permute(1, 2, 0, 3)  # (h, b, seq, dd)
    xpad = torch.cat([cached, xh], dim=2)
    assert torch.equal(out["cache"], xh[:, :, -L:, :])
    expect = (w @ xpad).permute(2, 1, 0, 3).reshape(seq, b, h * dd)
    assert torch.allclose(out["out"], expect, atol=1e-12)
    # tanh 门缺失会破坏数值——取块内闭式差异验证门控真实生效
    raw = xr[:, :, 1]
    assert not torch.allclose(xh, raw.permute(1, 2, 0, 3), atol=1e-3)


@check
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
    proj = x @ w.t() + bb  # (t, b, 2ch)
    a, g = proj[..., :ch], proj[..., ch:]
    expect = (a * torch.sigmoid(g)).permute(1, 2, 0)
    assert torch.allclose(out["gated"], expect, atol=1e-12)
    # mask 置零
    mask = torch.zeros(b, t, dtype=torch.bool)
    mask[0, 2] = True
    out_m = solve(mode="conv_gated", x=x, in_proj_w=w, in_proj_b=bb, mask=mask)
    assert torch.all(out_m["gated"][0, :, 2] == 0)
    assert torch.allclose(out_m["gated"][:, :, 0], expect[:, :, 0],
                          atol=1e-12)
    # in_proj 缺门（sigmoid 支路丢弃）会改变幅值——验证门真实参与
    assert not torch.allclose(out["gated"], a.permute(1, 2, 0), atol=1e-3)


@check
def check_chunk_scale(solve):
    """chunk_scale：1 + 头/尾修正，短块截断、长块补零。"""
    torch.set_default_dtype(torch.float64)
    ch, k = 3, 5
    scale = torch.randn(2, ch, k) * 0.5
    # chunk_size >= k：头修正贴左、尾修正贴右、中间为 1
    out = solve(mode="chunk_scale", scale=scale, chunk_size=8)
    assert out["scale"].shape == (ch, 8)
    assert torch.allclose(out["scale"][:, :5], 1.0 + scale[0], atol=1e-12)
    assert torch.allclose(out["scale"][:, -5:], 1.0 + scale[1], atol=1e-12)
    assert torch.allclose(out["scale"][:, 5:3], torch.zeros(ch, 0),
                          atol=1e-12)
    # chunk_size == k：两侧修正重叠区为相加
    out_k = solve(mode="chunk_scale", scale=scale, chunk_size=5)
    assert torch.allclose(out_k["scale"], 1.0 + scale[0] + scale[1],
                          atol=1e-12)
    # chunk_size < k：left 取前、right 取后
    out_s = solve(mode="chunk_scale", scale=scale, chunk_size=3)
    assert torch.allclose(out_s["scale"][:, :3],
                          1.0 + scale[0][:, :3] + scale[1][:, -3:],
                          atol=1e-12)


@check
def check_chunk_causal_conv(solve):
    """chunk 级因果卷积：整除恒等（逐块流式 == 一次性大块）、cache 滚动、
    causal/chunkwise 分解对拍。"""
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
    out = solve(mode="chunk_causal_conv", x=x, cache=cache, causal_w=causal_w,
                causal_b=causal_b, chunk_w=chunk_w, chunk_b=chunk_b,
                scale=scale, chunk_size=cs)
    assert out["out"].shape == x.shape
    # cache 滚动：cat 后取最后 left_pad
    assert torch.equal(out["cache"], torch.cat([cache, x], -1)[..., -left_pad:])
    # 分解对拍（单块 seq=cs 时 chunkwise 无跨块）
    xp = torch.cat([cache, x], -1)
    x_causal = F.conv1d(xp, causal_w, causal_b, groups=ch)
    x_chunk = F.conv1d(xp[..., left_pad:], chunk_w, chunk_b, groups=ch,
                       padding=0)
    cs_t = solve(mode="chunk_scale", scale=scale, chunk_size=cs)["scale"]
    assert torch.allclose(out["out"], x_chunk * cs_t.unsqueeze(0) + x_causal,
                          atol=1e-12)
    # 整除恒等：逐块流式拼接 == 一次性大块（chunk_size 相同）
    eq = solve(mode="stream_equiv", x=x, cache=cache, causal_w=causal_w,
               causal_b=causal_b, chunk_w=chunk_w, chunk_b=chunk_b,
               scale=scale, chunk_size=cs)
    whole_pieces = torch.cat(eq["outs"], dim=-1)
    assert torch.allclose(whole_pieces, eq["whole"], atol=1e-12)
    assert torch.allclose(eq["caches"][-1],
                          torch.cat([cache, x], -1)[..., -left_pad:],
                          atol=1e-12)
    # 双块走账：第二块的 cache 来自第一块的尾部
    r1 = solve(mode="chunk_causal_conv", x=x[..., :cs], cache=cache,
               causal_w=causal_w, causal_b=causal_b, chunk_w=chunk_w,
               chunk_b=chunk_b, scale=scale, chunk_size=cs)
    r2 = solve(mode="chunk_causal_conv", x=x[..., cs:], cache=r1["cache"],
               causal_w=causal_w, causal_b=causal_b, chunk_w=chunk_w,
               chunk_b=chunk_b, scale=scale, chunk_size=cs)
    assert torch.allclose(torch.cat([r1["out"], r2["out"]], -1),
                          eq["whole"], atol=1e-12)
    # causal 支路是纯因果的：改 x 的未来帧不影响第一块 x_causal 部分
    x_mut = x.clone()
    x_mut[..., cs:] += 1.0
    r1m = solve(mode="chunk_causal_conv", x=x_mut[..., :cs], cache=cache,
                causal_w=causal_w, causal_b=causal_b, chunk_w=chunk_w,
                chunk_b=chunk_b, scale=scale, chunk_size=cs)
    assert torch.allclose(r1["out"], r1m["out"], atol=1e-12)
    # chunkwise 支路在 chunk_size=1 时缩为仅左邻（k=5 时窗口截到 1）
    out1 = solve(mode="chunk_causal_conv", x=x[..., :1], cache=cache,
                 causal_w=causal_w, causal_b=causal_b, chunk_w=chunk_w,
                 chunk_b=chunk_b, scale=scale, chunk_size=1)
    s1 = solve(mode="chunk_scale", scale=scale, chunk_size=1)["scale"]
    x_c1 = F.conv1d(torch.cat([cache, x[..., :1]], -1), causal_w, causal_b,
                    groups=ch)
    x_g1 = (F.conv1d(torch.cat([cache, x[..., :1]], -1)[..., left_pad:],
                     chunk_w, chunk_b, groups=ch, padding=0)
            * s1.unsqueeze(0))
    assert torch.allclose(out1["out"], x_g1 + x_c1, atol=1e-12)


def check(solve):
    for fn in CHECKS:
        fn(solve)
