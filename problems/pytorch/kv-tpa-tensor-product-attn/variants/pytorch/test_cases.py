import math

import numpy as np
import torch

KEYS = {"out", "prob"}


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _oracle(aq, ak, av, bq, bk, bv, is_causal=False, attn_mask=None,
            scale=None, scale_q=None, scale_k=None, scale_v=None, gate=True):
    aq = aq.detach().numpy().astype(np.float64)
    ak = ak.detach().numpy().astype(np.float64)
    av = av.detach().numpy().astype(np.float64)
    bq = bq.detach().numpy().astype(np.float64)
    bk = bk.detach().numpy().astype(np.float64)
    bv = bv.detach().numpy().astype(np.float64)
    batch, n, heads, rank = aq.shape
    m, s = ak.shape[1], ak.shape[3]
    s_v = av.shape[3]
    d = bq.shape[3]
    e = bv.shape[2]

    if scale is None:
        scale = d ** -0.5
    if scale_q is None:
        scale_q = 1.0 / rank
    if scale_k is None:
        scale_k = 1.0 / s
    if scale_v is None:
        scale_v = 1.0 / s_v

    if gate:
        aq, ak, av = _sig(aq), _sig(ak), _sig(av)
    q = np.einsum("bnhr,bnrd->bnhd", aq, bq) * scale_q
    k = np.einsum("bmhs,bmds->bmhd", ak, bk) * scale_k
    v = np.einsum("bmhs,bmes->bmhe", av, bv) * scale_v

    bias = np.zeros((n, m), dtype=np.float64)
    if is_causal:
        bias[np.triu(np.ones((n, m), dtype=bool), 1)] = -np.inf
    if attn_mask is not None:
        mask = attn_mask.detach().numpy()
        if mask.dtype == np.bool_:
            bias[~mask] = -np.inf
        else:
            bias = bias + mask.astype(np.float64)

    score = np.einsum("bnhd,bmhd->bhnm", q, k) * scale + bias
    score = score - score.max(axis=-1, keepdims=True)
    exp = np.exp(score)
    prob = exp / exp.sum(axis=-1, keepdims=True)
    out = np.einsum("bhnm,bmhe->bnhe", prob, v)
    return torch.tensor(out), torch.tensor(prob)


def _close(got, want, dtype):
    tol = 3e-5 if dtype == torch.float32 else 2e-9
    torch.testing.assert_close(got, want, rtol=tol, atol=tol)


def _rand(shape, dtype, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=gen, dtype=dtype)


def _verify(solve, aq, ak, av, bq, bk, bv, **kwargs):
    dtype = aq.dtype
    before = [t.clone() for t in (aq, ak, av, bq, bk, bv)]
    got = solve(aq, ak, av, bq, bk, bv, **kwargs)
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    for t, old in zip((aq, ak, av, bq, bk, bv), before):
        assert torch.equal(t, old), "输入被就地修改了"
    exp_shape = (aq.shape[0], aq.shape[1], aq.shape[2], bv.shape[2])
    assert got["out"].shape == exp_shape, got["out"].shape
    assert got["prob"].shape == (aq.shape[0], aq.shape[2], aq.shape[1], ak.shape[1])
    assert got["out"].dtype == dtype and got["prob"].dtype == dtype
    want_out, want_prob = _oracle(aq, ak, av, bq, bk, bv, **kwargs)
    _close(got["out"], want_out.to(dtype), dtype)
    _close(got["prob"], want_prob.to(dtype), dtype)
    return got


def _toy(dtype=torch.float64):
    aq = torch.tensor([[[[0.0]], [[0.0]]]], dtype=dtype)
    bq = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]], dtype=dtype)
    ak = torch.zeros(1, 3, 1, 1, dtype=dtype)
    bk = torch.tensor([[[[1.0], [0.0]], [[0.0], [1.0]], [[1.0], [1.0]]]], dtype=dtype)
    av = torch.zeros(1, 3, 1, 1, dtype=dtype)
    bv = torch.tensor([[[[1.0], [0.0]], [[0.0], [1.0]], [[2.0], [2.0]]]], dtype=dtype)
    return aq, ak, av, bq, bk, bv


def check_example(solve):
    """题面示例：sigmoid(0)=0.5、causal、默认 scale_q=1/R、scale_k=1/S、scale_v=1/S_V。"""
    aq, ak, av, bq, bk, bv = _toy()
    got = _verify(solve, aq, ak, av, bq, bk, bv, is_causal=True)
    out = got["out"].tolist()
    prob = got["prob"].tolist()
    assert prob[0][0][0] == [1.0, 0.0, 0.0], prob
    expected = [0.45592056, 0.54407944, 0.0]
    for x, y in zip(prob[0][0][1], expected):
        assert abs(x - y) < 1e-7, (prob[0][0][1], expected)
    assert out[0][0][0] == [0.5, 0.0], out
    assert abs(out[0][1][0][0] - 0.22796028) < 1e-7, out
    assert abs(out[0][1][0][1] - 0.27203972) < 1e-7, out


check_example.example = True


def check_matches_oracle(solve):
    """多组形状/掩码组合对拍 numpy 独立实现（sigmoid 门 + 默认/自定义 scale）。"""
    configs = [
        dict(shape=(1, 1, 4, 2, 3, 2, 2, 2, 2), causal=False),
        dict(shape=(2, 3, 5, 1, 4, 2, 3, 2, 3), causal=True),
        dict(shape=(2, 2, 3, 2, 2, 2, 2, 2, 2), causal=False),
        dict(shape=(1, 3, 6, 2, 3, 2, 2, 4, 3), causal=True),
        dict(shape=(3, 1, 2, 3, 2, 4, 4, 2, 5), causal=False),
    ]
    for i, cfg in enumerate(configs):
        b, n, m, h, r, d, s, sv, e = cfg["shape"]
        dtype = torch.float32 if i % 2 else torch.float64
        aq = _rand((b, n, h, r), dtype, 10 + i)
        ak = _rand((b, m, h, s), dtype, 20 + i)
        av = _rand((b, m, h, sv), dtype, 30 + i)
        bq = _rand((b, n, r, d), dtype, 40 + i)
        bk = _rand((b, m, d, s), dtype, 50 + i)
        bv = _rand((b, m, e, sv), dtype, 60 + i)
        _verify(solve, aq, ak, av, bq, bk, bv, is_causal=cfg["causal"])


def check_sigmoid_gate_is_required(solve):
    """aq/ak/av 必须先过 sigmoid：跳过它点出来的结果对不上，且能被区分。"""
    aq, ak, av, bq, bk, bv = _toy()
    aq = aq + 2.0
    ak = ak - 1.5
    av = av + 0.75
    got = _verify(solve, aq, ak, av, bq, bk, bv)
    want_gate, _ = _oracle(aq, ak, av, bq, bk, bv, gate=True)
    want_raw, _ = _oracle(aq, ak, av, bq, bk, bv, gate=False)
    assert not torch.allclose(got["out"], want_raw, rtol=1e-5, atol=1e-5), (
        "sigmoid 门似乎被跳过了")
    _close(got["out"], want_gate.to(aq.dtype), aq.dtype)


def check_scales_are_required(solve):
    """默认 scale_q=1/R、scale_k=1/S、scale_v=1/S_V、scale=D^-0.5，自定义值也必须生效。"""
    dtype = torch.float64
    b, n, m, h, r, d, s, sv, e = 2, 3, 4, 2, 3, 5, 2, 4, 3
    aq = _rand((b, n, h, r), dtype, 31)
    ak = _rand((b, m, h, s), dtype, 32)
    av = _rand((b, m, h, sv), dtype, 33)
    bq = _rand((b, n, r, d), dtype, 34)
    bk = _rand((b, m, d, s), dtype, 35)
    bv = _rand((b, m, e, sv), dtype, 36)

    got = solve(aq, ak, av, bq, bk, bv)
    want_default, _ = _oracle(aq, ak, av, bq, bk, bv)
    _close(got["out"], want_default, dtype)

    # 跳过 scale_q 或 scale_k 会改变结果，确保默认值真的被应用
    want_no_q, _ = _oracle(aq, ak, av, bq, bk, bv, scale_q=1.0)
    assert not torch.allclose(got["out"], want_no_q, rtol=1e-6, atol=1e-6), (
        "scale_q 的 1/R 默认值没有生效")
    want_no_k, _ = _oracle(aq, ak, av, bq, bk, bv, scale_k=1.0)
    assert not torch.allclose(got["out"], want_no_k, rtol=1e-6, atol=1e-6), (
        "scale_k 的 1/S 默认值没有生效")
    want_no_v, _ = _oracle(aq, ak, av, bq, bk, bv, scale_v=1.0)
    assert not torch.allclose(got["out"], want_no_v, rtol=1e-6, atol=1e-6), (
        "scale_v 的 1/S_V 默认值没有生效")

    custom = dict(scale=0.7, scale_q=0.25, scale_k=1.5, scale_v=2.0)
    got_custom = solve(aq, ak, av, bq, bk, bv, **custom)
    want_custom, _ = _oracle(aq, ak, av, bq, bk, bv, **custom)
    _close(got_custom["out"], want_custom, dtype)


def check_causal_rectangular(solve):
    """N!=M 时 causal 是左上对齐的矩形 tril：第 i 行只看 j<=i。"""
    dtype = torch.float64
    aq = _rand((1, 3, 2, 2), dtype, 1)
    ak = _rand((1, 5, 2, 2), dtype, 2)
    av = _rand((1, 5, 2, 2), dtype, 3)
    bq = _rand((1, 3, 2, 3), dtype, 4)
    bk = _rand((1, 5, 3, 2), dtype, 5)
    bv = _rand((1, 5, 4, 2), dtype, 6)
    got = _verify(solve, aq, ak, av, bq, bk, bv, is_causal=True)
    prob = got["prob"]
    for i in range(3):
        for j in range(5):
            if j > i:
                assert prob[:, :, i, j].abs().max().item() == 0.0, (i, j, prob)
        assert abs(prob[:, :, i, :].sum(dim=-1) - 1.0).max().item() < 1e-9
    assert (prob[:, :, 0, 0] == 1.0).all()


def check_bool_and_float_masks_agree(solve):
    """bool mask（True=保留）与等价的 0/-inf 加法 mask 结果一致；被遮位置概率恰为 0。"""
    dtype = torch.float64
    aq = _rand((2, 4, 2, 3), dtype, 7)
    ak = _rand((2, 4, 2, 2), dtype, 8)
    av = _rand((2, 4, 2, 2), dtype, 9)
    bq = _rand((2, 4, 3, 3), dtype, 11)
    bk = _rand((2, 4, 3, 2), dtype, 12)
    bv = _rand((2, 4, 3, 2), dtype, 13)

    keep = torch.ones((4, 4), dtype=torch.bool).tril(0)
    keep[1, 0] = False
    keep[3, 2] = False

    got_bool = _verify(solve, aq, ak, av, bq, bk, bv, attn_mask=keep)
    additive = torch.zeros((4, 4), dtype=dtype)
    additive[~keep] = float("-inf")
    got_float = _verify(solve, aq, ak, av, bq, bk, bv, attn_mask=additive)
    torch.testing.assert_close(got_bool["out"], got_float["out"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(got_bool["prob"], got_float["prob"], rtol=0.0, atol=0.0)

    assert got_bool["prob"][:, :, 1, 0].abs().max().item() == 0.0
    assert got_bool["prob"][:, :, 3, 2].abs().max().item() == 0.0
    for i in range(4):
        assert abs(got_bool["prob"][:, :, i, :].sum(dim=-1) - 1.0).max().item() < 1e-9


def check_shapes_and_no_mutation(solve):
    """输出形状/dtype、概率行和为 1，且不修改任何输入。"""
    dtype = torch.float32
    aq = _rand((2, 2, 3, 2), dtype, 21)
    ak = _rand((2, 7, 3, 2), dtype, 22)
    av = _rand((2, 7, 3, 4), dtype, 23)
    bq = _rand((2, 2, 2, 5), dtype, 24)
    bk = _rand((2, 7, 5, 2), dtype, 25)
    bv = _rand((2, 7, 6, 4), dtype, 26)
    got = _verify(solve, aq, ak, av, bq, bk, bv)
    assert got["out"].shape == (2, 2, 3, 6)
    assert got["prob"].shape == (2, 3, 2, 7)
    assert got["out"].dtype == torch.float32 and got["prob"].dtype == torch.float32
    assert abs(got["prob"].sum(dim=-1) - 1.0).max().item() < 3e-5


def check_is_causal_excludes_mask(solve):
    """is_causal=True 与 attn_mask 同时给出必须报错（上游 assert attn_mask is None）。"""
    aq, ak, av, bq, bk, bv = _toy()
    mask = torch.ones((2, 3), dtype=torch.bool)
    try:
        solve(aq, ak, av, bq, bk, bv, is_causal=True, attn_mask=mask)
    except ValueError:
        return
    raise AssertionError("is_causal 与 attn_mask 同给时应抛 ValueError")


def check_validation(solve):
    """维度不一致、dtype 混杂、mask 形状/类型错误、非法 scale 都要抛 ValueError。"""
    aq, ak, av, bq, bk, bv = _toy()

    def bad(fn):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError("应抛 ValueError")

    bad(lambda: solve(aq, ak, av, bq.repeat(1, 1, 2, 1), bk, bv))
    bad(lambda: solve(aq.float(), ak, av, bq, bk, bv))
    bad(lambda: solve(aq, ak[:, :2], av[:, :2], bq, bk, bv))
    bad(lambda: solve(aq, ak, av, bq, bk.to(torch.float32), bv))
    bad(lambda: solve(aq, ak[0], av, bq, bk, bv))
    bad(lambda: solve(aq, ak, av, bq, bk, bv, attn_mask=torch.ones(3, 3, dtype=torch.bool)))
    bad(lambda: solve(aq, ak, av, bq, bk, bv, attn_mask=torch.ones(2, 3, dtype=torch.int64)))
    bad(lambda: solve(aq, ak, av, bq, bk, bv, scale=0.0, scale_q=True))
    bad(lambda: solve(aq, ak, av, bq, bk, bv, is_causal=1))
    bad(lambda: solve(aq, ak, av, bq, bk, bv, scale=float("nan")))
