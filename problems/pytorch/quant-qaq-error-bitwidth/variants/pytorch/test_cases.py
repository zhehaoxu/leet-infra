"""独立 numpy/标量 oracle + 针对性检查，不复用 solution 的实现。"""

import bisect
import math
from statistics import NormalDist

import numpy as np
import torch

KEYS = ("outlier_mask", "max_error", "n_bits_clamped", "n_bits", "quant_dequant")
NQ = {"token": 3, "layer": 2, "head": 1}


def _mask(block, ratio):
    """kthvalue 双阈值 + 非对称比较：低端 <= 进掩码，高端 > 才进。"""
    order = sorted(range(len(block)), key=lambda i: (block[i], i))
    low = block[order[int(math.ceil(ratio / 2 * len(block))) - 1]]
    high = block[order[int(math.floor((1.0 - ratio / 2) * len(block))) - 1]]
    return [(v <= low) or (v > high) for v in block]


def _tables(n_max):
    """scipy.stats.norm.ppf 的替身：NormalDist 求分位，上界表尾部补 +inf。"""
    nd = NormalDist()
    up = tuple(torch.tensor([nd.inv_cdf((i + 1) / 2 ** n) for i in range(2 ** n - 1)] + [math.inf],
                            dtype=torch.float64) for n in range(n_max + 1))
    mid = tuple(torch.tensor([nd.inv_cdf((i + 0.5) / 2 ** n) for i in range(2 ** n)],
                             dtype=torch.float64) for n in range(n_max + 1))
    return up, mid


def _oracle(cache, kv, level, sym, ratio, last_n, target, nmin, nmax,
            q_norm=None, att=None, method="uniform", up=None, mid=None):
    x = np.asarray(cache, dtype=np.float64)
    nq = NQ[level]
    lead, block = x.shape[:-nq], x.shape[-nq:]
    m, seq_len = int(np.prod(block)), x.shape[1]
    mask, raw, final, out = np.zeros(x.shape, bool), np.zeros(lead, np.int64), \
        np.zeros(lead, np.int64), x.copy()
    err = np.zeros(lead)
    for unit in np.ndindex(*lead):
        blk = x[unit].reshape(-1)
        bad = [False] * m if ratio == 0.0 else _mask(blk, ratio)
        kept = [v for v, b in zip(blk, bad) if not b]
        mu = math.fsum(kept) / len(kept)
        mean = 0.0 if sym else mu
        span = 2.0 * max(abs(v) for v in kept) if sym else max(kept) - min(kept)
        if kv == "key":
            me = math.sqrt(12.0 / q_norm * math.log(seq_len ** 3 / (seq_len - 1) * target ** 2 + 1))
        else:
            a_all = np.asarray(att, dtype=np.float64)
            a = a_all[:, unit[0], :, a_all.shape[3] - last_n:, unit[1]]
            pick = a if level == "token" else (
                a[unit[2]] if level == "layer" else a[unit[2], unit[3]])
            me = math.sqrt(12.0 / seq_len) * target / float(np.max(pick))
        err[unit] = me
        n = int(min(max(math.ceil(math.log2(span / (2.0 * me) + 1)), nmin), nmax))
        raw[unit] = n
        if last_n > 1 and unit[1] >= seq_len - last_n + 1:
            n = nmax
        final[unit] = n
        if method == "uniform":
            scale, lo, hi = span / 2.0 ** n, -(2.0 ** (n - 1)), 2.0 ** (n - 1) - 1.0
        else:
            scale = math.sqrt(math.fsum((v - mu) ** 2 for v in kept) / (len(kept) - 1))
            edges = [float(t) for t in up[n]]
        vals = [v if b else (min(max(round((v - mean) / scale), lo), hi) * scale + mean
                             if method == "uniform" else
                             float(mid[n][bisect.bisect_left(edges, (v - mean) / scale)]) * scale + mean)
                for v, b in zip(blk, bad)]
        mask[unit] = np.array(bad, bool).reshape(block)
        out[unit] = np.array(vals).reshape(block)
    return mask, err, raw, final, out


def _verify(solve, args, kw):
    cache, before = args[0], args[0].clone()
    got = solve(*args, **kw)
    assert isinstance(got, dict) and set(got) == set(KEYS), f"返回键不对：{sorted(got)}"
    assert torch.equal(cache, before), "输入被就地修改了"
    lead, m = tuple(cache.shape[: 5 - NQ[kw["level"]]]), got["outlier_mask"]
    assert m.dtype == torch.bool and tuple(m.shape) == tuple(cache.shape)
    assert tuple(got["n_bits"].shape) == lead == tuple(got["n_bits_clamped"].shape)
    assert got["n_bits"].dtype == torch.int64 and got["n_bits_clamped"].dtype == torch.int64
    assert got["quant_dequant"].shape == cache.shape and got["max_error"].dtype == cache.dtype
    assert tuple(got["max_error"].shape) == lead, "max_error 要广播到每个量化单元"
    assert got["quant_dequant"].dtype == cache.dtype
    assert kw["n_bits_min"] <= int(got["n_bits"].min()) <= int(got["n_bits"].max()) <= kw["n_bits_max"]
    assert torch.equal(got["quant_dequant"][m], cache[m]), "outlier 位置必须保留原值"
    mask, err, raw, final, out = _oracle(
        cache.numpy(), kw["key_or_value"], kw["level"], kw["symmetric"], kw["outliers_ratio"],
        kw["last_n_attentions"], kw["target_quantization_error"], kw["n_bits_min"],
        kw["n_bits_max"], kw.get("q_norm"), None if args[1] is None else args[1].numpy(),
        kw.get("method", "uniform"), kw.get("quantile_upper"), kw.get("quantile_center"))
    assert torch.equal(got["outlier_mask"], torch.from_numpy(mask)), "outlier 掩码不对"
    assert got["n_bits_clamped"].tolist() == raw.tolist(), got["n_bits_clamped"].tolist()
    assert got["n_bits"].tolist() == final.tolist(), got["n_bits"].tolist()
    tol = 3e-5 if cache.dtype == torch.float32 else 2e-9
    torch.testing.assert_close(got["max_error"], torch.as_tensor(err, dtype=cache.dtype),
                               rtol=tol, atol=tol)
    torch.testing.assert_close(got["quant_dequant"], torch.tensor(out, dtype=cache.dtype),
                               rtol=tol, atol=tol)
    return got


def _example():
    cache = torch.tensor([[-3.0, -0.5, 1.0, 4.0], [-2.0, 0.25, 0.5, 6.0],
                          [-1.0, -0.25, 0.5, 3.0]], dtype=torch.float64)
    return (cache.reshape(1, 3, 1, 1, 4), None), dict(
        key_or_value="key", level="token", symmetric=True, outliers_ratio=0.5,
        last_n_attentions=2, target_quantization_error=0.01, n_bits_min=1, n_bits_max=4, q_norm=3.0)


def check_example(solve):
    """题面示例：token 级、outliers_ratio=0.5、末 1 个 token 被抬到 n_bits_max。"""
    got = _verify(solve, *_example())
    assert got["outlier_mask"].reshape(3, 4).tolist() == [[True, False, False, True]] * 3
    assert got["n_bits_clamped"].tolist() == [[4, 3, 3]], got["n_bits_clamped"].tolist()
    assert got["n_bits"].tolist() == [[4, 3, 4]], got["n_bits"].tolist()
    assert abs(float(got["max_error"].max()) - 0.0734599093) < 1e-9, got["max_error"].tolist()
    assert got["max_error"].min() == got["max_error"].max(), "key 分支的误差与 cache 无关，逐单元同值"
    assert got["quant_dequant"].reshape(3, 4).tolist() == [
        [-3.0, -0.5, 0.875, 4.0], [-2.0, 0.25, 0.375, 6.0], [-1.0, -0.25, 0.4375, 3.0]]


check_example.example = True


def check_mask_and_ceil(solve):
    """掩码低端 <= / 高端 >；位宽是 ceil(log2(scale/(2e)+1))，不是四舍五入。"""
    cache = torch.tensor([[1.0, 1.0, 1.0, 2.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]]
                         ).reshape(1, 2, 1, 1, 5)
    got = _verify(solve, (cache, None), dict(_example()[1], outliers_ratio=0.4, last_n_attentions=1))
    assert got["outlier_mask"].reshape(2, 5).tolist() == [
        [True, True, True, False, True], [True, False, False, False, True]], "下阈值用 <=，上阈值用 >"
    row = torch.tensor([[1.0, -2.0, 0.5, 3.0]] * 2).reshape(1, 2, 1, 1, 4)
    log_term = math.log(2 ** 3 * 0.01 ** 2 + 1)
    for frac, expect in ((4.2, 5), (3.2, 4)):
        d = 2.0 ** frac - 1.0                    # scale/(2*max_error)，raw 恰为 frac
        got = _verify(solve, (row, None), dict(_example()[1], outliers_ratio=0.0,
                     last_n_attentions=1, n_bits_min=1, n_bits_max=8,
                     q_norm=4.0 * d ** 2 * 12.0 * log_term / 6.0 ** 2))
        assert got["n_bits"].tolist() == [[expect, expect]], (frac, got["n_bits"].tolist())


def check_clamp_and_trailing(solve):
    """raw<0 被 clamp 到 n_bits_min；1 bit 只能表示 [-scale, 0]；末 last_n-1 行无条件抬到 max。"""
    cache = torch.tensor([[1.0, -1.0, 0.5, -0.5]] * 2).reshape(1, 2, 1, 1, 4)
    got = _verify(solve, (cache, None), dict(_example()[1], outliers_ratio=0.0,
                 last_n_attentions=1, n_bits_min=1, n_bits_max=6, target_quantization_error=100.0))
    assert got["n_bits"].tolist() == [[1, 1]], got["n_bits"].tolist()
    assert got["quant_dequant"].reshape(-1).tolist()[:4] == [0.0, -1.0, 0.0, -0.0], (
        "1 bit 对称量化 clamp 到 [-1, 0]，正数一律塌到 0")
    rng = np.random.default_rng(7)
    big = torch.tensor(rng.normal(size=(2, 6, 2, 2, 8)) * 2.0)
    base = dict(_example()[1], outliers_ratio=0.25, n_bits_min=2, n_bits_max=5)
    plain = _verify(solve, (big, None), dict(base, last_n_attentions=1))
    assert torch.equal(plain["n_bits"], plain["n_bits_clamped"])
    for last_n in (2, 4):
        got = _verify(solve, (big, None), dict(base, last_n_attentions=last_n))
        tail = 6 - last_n + 1
        assert (got["n_bits"][:, tail:] == 5).all(), "末 last_n-1 个 token 必须抬到 n_bits_max"
        assert torch.equal(got["n_bits_clamped"], plain["n_bits_clamped"])
        assert torch.equal(got["n_bits"][:, :tail], plain["n_bits"][:, :tail]), "只有末几行被覆盖"


def check_value_branch_uses_last_n_attentions(solve):
    """value 分支只看 attention 末 last_n 行；更早的历史行改了不影响位宽。"""
    rng = np.random.default_rng(13)
    cache = torch.tensor(rng.normal(size=(1, 4, 1, 1, 8)) * 1.5)
    att = rng.uniform(0.05, 0.9, size=(1, 1, 1, 4, 4))
    kw = dict(_example()[1], key_or_value="value", outliers_ratio=0.25, last_n_attentions=2,
              n_bits_min=1, n_bits_max=7)
    base = _verify(solve, (cache, torch.tensor(att)), kw)
    assert tuple(base["max_error"].shape) == (1, 4)
    older = att.copy()
    older[:, :, :, :-2, :] = 0.99
    assert torch.equal(_verify(solve, (cache, torch.tensor(older)), kw)["n_bits"], base["n_bits"]), (
        "last_n 之外的历史行不该影响位宽")
    newer = att.copy()
    newer[:, :, :, -2:, :] = 0.02
    moved = _verify(solve, (cache, torch.tensor(newer)), kw)
    assert (moved["max_error"] > base["max_error"]).all()
    assert (moved["n_bits"] <= base["n_bits"]).all() and not torch.equal(moved["n_bits"], base["n_bits"])


def check_normal_method_uses_searchsorted(solve):
    """normal：std 归一化 + searchsorted 落桶 + 取桶中心；symmetric 只把中心化均值置 0。"""
    cache = torch.tensor([[-2.0, -0.4, 0.3, 2.5], [0.1, 0.2, -0.3, 0.5]],
                         dtype=torch.float64).reshape(1, 2, 1, 1, 4)
    upper, mid = _tables(4)
    kw = dict(_example()[1], outliers_ratio=0.0, last_n_attentions=1, n_bits_min=2, n_bits_max=4,
              method="normal", quantile_upper=upper, quantile_center=mid)
    got = _verify(solve, (cache, None), kw)
    for row in range(2):
        x = cache[0, row].reshape(-1).tolist()
        mu = math.fsum(x) / len(x)
        std = math.sqrt(math.fsum((v - mu) ** 2 for v in x) / (len(x) - 1))
        edges = [float(t) for t in upper[int(got["n_bits"][0, row])]]
        want = [float(mid[int(got["n_bits"][0, row])][bisect.bisect_left(edges, v / std)]) * std
                for v in x]
        assert np.allclose(got["quant_dequant"][0, row].tolist(), want, rtol=0, atol=1e-9), (
            got["quant_dequant"][0, row].tolist(), want, "symmetric 只把中心化均值置 0，std 仍围绕真实均值")
    masked_run = _verify(solve, (cache, None), dict(kw, outliers_ratio=0.5))
    masked = masked_run["outlier_mask"]
    assert int(masked.sum()) > 0, "0.5 的 outlier 比例在该 cache 上应当真的选出一些元素"
    assert torch.equal(masked_run["quant_dequant"][masked], cache[masked])


def check_dtype_and_errors(solve):
    """float32 入 float32 出；非法输入一律 ValueError。"""
    rng = np.random.default_rng(17)
    x = rng.normal(size=(1, 4, 1, 2, 4)) * 2.0
    kw = dict(_example()[1], level="head", outliers_ratio=0.25, last_n_attentions=2,
              n_bits_min=2, n_bits_max=5)
    lo, hi = solve(torch.tensor(x, dtype=torch.float32), None, **kw), solve(torch.tensor(x), None, **kw)
    assert lo["quant_dequant"].dtype == torch.float32 and lo["n_bits"].dtype == torch.int64
    torch.testing.assert_close(lo["quant_dequant"].double(), hi["quant_dequant"], rtol=1e-5, atol=1e-5)
    assert lo["n_bits"].tolist() == hi["n_bits"].tolist()
    good = dict(key_or_value="key", level="token", symmetric=True, outliers_ratio=0.25,
                last_n_attentions=1, target_quantization_error=0.01, n_bits_min=1,
                n_bits_max=4, q_norm=2.0)
    c, att = torch.tensor(x), torch.tensor(rng.uniform(0.1, 0.9, size=(2, 1, 2, 2, 4)))
    bad = [(c, None, dict(good, level="block")), (c, None, dict(good, key_or_value="both")),
           (c, None, dict(good, q_norm=None)), (c[0], None, good), (c.to(torch.int32), None, good),
           (c, None, dict(good, outliers_ratio=1.0)), (c, None, dict(good, n_bits_min=5)),
           (c, None, dict(good, last_n_attentions=0)), (c, None, dict(good, target_quantization_error=0.0)),
           (c, None, dict(good, key_or_value="value")), (c, att[:, :, :, :, :3], dict(good, key_or_value="value")),
           (c, att, dict(good, key_or_value="value", method="normal"))]
    for i, (cache, at, kw_bad) in enumerate(bad):
        try:
            solve(cache, at, **kw_bad)
        except ValueError:
            continue
        raise AssertionError(f"第 {i} 组非法输入应抛 ValueError：{kw_bad}")


def check_random_against_oracle(solve):
    """30 组随机实例（两分支、三 level、两方法、两对称性）+ 非连续输入与标量 oracle 对拍。"""
    rng = np.random.default_rng(20261101)
    upper, mid = _tables(6)
    for case in range(30):
        shape = (1 + case % 2, 6, 1 + case % 2, 2, 8)
        cache = torch.tensor(rng.normal(size=shape) * (0.5 + case % 3))
        att = torch.tensor(rng.uniform(0.05, 0.95, size=(shape[2], shape[0], shape[3], 3, shape[1])))
        value = case % 4 >= 2
        kw = dict(key_or_value="value" if value else "key", level=["token", "layer", "head"][case % 3],
                  symmetric=bool(case % 2), outliers_ratio=[0.0, 0.25, 0.5][case % 3],
                  last_n_attentions=1 + case % 3, target_quantization_error=0.01,
                  n_bits_min=1 + case % 2, n_bits_max=4 + case % 3, q_norm=1.5 + 0.5 * (case % 4))
        if case % 5 == 0:
            kw.update(method="normal", quantile_upper=upper, quantile_center=mid)
        _verify(solve, (cache, att if value else None), kw)
    big = torch.tensor(rng.normal(size=(3, 6, 1, 1, 8)) * 2.0)
    a = solve(big[::2], None, "key", "token", True, 0.25, 2, 0.01, 1, 5, q_norm=2.0)
    b = solve(big[::2].contiguous(), None, "key", "token", True, 0.25, 2, 0.01, 1, 5, q_norm=2.0)
    assert all(torch.equal(a[k], b[k]) for k in KEYS), "非连续输入的结果必须与连续副本一致"
