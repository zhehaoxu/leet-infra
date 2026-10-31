"""Independent pure-Python oracle (scalar loops, math.floor/log/exp) plus targeted checks."""

import math

import torch

K0, K1, EPS = -0.1, 1.1, 1e-6
KEYS = ("q", "r", "h", "alpha", "betas", "soft_loss", "searched_weight", "chosen_offset",
        "searched_mse")


def _decode(h, res, log_res):
    """Scalar twin of the ladder decode: floor -> clamp(eps) -> sigmoid((logit + log res)/2)."""
    ladder = math.floor(h * res) / res
    ladder = min(max(ladder, EPS), 1.0 - EPS)
    logit = math.log(ladder) - math.log(1.0 - ladder)
    z = -(logit + log_res) / 2.0
    return 1.0 / (1.0 + math.exp(z))


def _round_half_even(value):
    """torch.round 的口径：就近取整、.5 取偶。"""
    low = math.floor(value)
    gap = value - low
    if gap > 0.5:
        return float(low + 1)
    if gap < 0.5:
        return float(low)
    return float(low) if low % 2 == 0 else float(low + 1)


def _score(alpha, z, X, scale, rows, cols, beta, power):
    err = 0.0
    for o in range(rows):
        for t in range(len(X)):
            acc = 0.0
            for j in range(cols):
                acc += X[t][j] * ((alpha[o][j] - z[o][j]) * scale)
            err += acc * acc
    reg = 0.0
    for o in range(rows):
        for j in range(cols):
            reg += abs(alpha[o][j] - 0.5) ** power
    return err / cols + beta * reg / (rows * cols)


def _oracle(W, X, scale, num_iters, steps, lam):
    """纯 Python 标量实现：逐元素、逐 (o, t, j) 累加，与参考解不共享控制流。"""
    if int(num_iters) < 1:
        raise ValueError("num_iters must be >= 1")
    if int(steps) < 1:
        raise ValueError("steps must be >= 1")
    if float(scale) <= 0.0:
        raise ValueError("scale must be > 0")
    if not (float(lam) > 0.0 and float(lam) != 1.0):
        raise ValueError("lam must be > 0 and != 1")
    if W.dim() != 2 or X.dim() != 2 or W.shape[1] != X.shape[1]:
        raise ValueError("W must be 2-D, X must be 2-D and share W's input channels")
    if W.dtype != X.dtype:
        raise ValueError("W and X must have the same dtype")
    if min(int(W.shape[0]), int(W.shape[1]), int(X.shape[0])) < 1:
        raise ValueError("empty W / X is not supported")

    dtype = W.dtype if W.dtype in (torch.float32, torch.float64) else torch.float32
    wl = W.detach().to(torch.float64).contiguous().tolist()
    xl = X.detach().to(torch.float64).contiguous().tolist()
    rows, cols = len(wl), len(wl[0])
    total = int(steps)
    scale = float(scale)
    res = float(2 ** int(num_iters))
    log_res = math.log(res)
    log_lam = math.log(float(lam))
    offsets = (0.0, 1.0 / res, -1.0 / res, 1.0, -1.0)

    q = [[math.floor(w / scale) for w in row] for row in wl]
    r = [[wl[o][j] - q[o][j] * scale for j in range(cols)] for o in range(rows)]
    z = [[r[o][j] / scale for j in range(cols)] for o in range(rows)]
    h = [[1.0 if q[o][j] == 0 else min(1.0, max(0.0, (z[o][j] - K0) / (K1 - K0)))
          for j in range(cols)] for o in range(rows)]

    chosen = [[0.0] * cols for _ in range(rows)]
    frozen = [[False] * cols for _ in range(rows)]
    soft = []
    for i in range(total):
        power = 1.0 + (i + 1) / total
        beta = power / log_lam
        for k in range(rows * cols):
            o, j = divmod(k, cols)
            if frozen[o][j]:
                continue
            best = None
            best_off = 0.0
            for off in offsets:
                trial = [row[:] for row in h]
                trial[o][j] = min(1.0, max(0.0, h[o][j] + off))
                alpha = [[_decode(trial[a][b], res, log_res) for b in range(cols)]
                         for a in range(rows)]
                value = _score(alpha, z, xl, scale, rows, cols, beta, power)
                if best is None or value < best:
                    best = value
                    best_off = off
            h[o][j] = min(1.0, max(0.0, h[o][j] + best_off))
            chosen[o][j] = best_off
            frozen[o][j] = abs(h[o][j] - 0.5) > 0.5 / res
        cur = [[_decode(h[a][b], res, log_res) for b in range(cols)] for a in range(rows)]
        soft.append(_score(cur, z, xl, scale, rows, cols, beta, power))

    alpha = [[_decode(h[a][b], res, log_res) for b in range(cols)] for a in range(rows)]
    searched = [[0.0] * cols for _ in range(rows)]
    mse = 0.0
    for o in range(rows):
        for j in range(cols):
            if abs(h[o][j] - 0.5) <= 0.5 / res + EPS:
                hard = 1.0 if (math.floor(h[o][j] * res) / res) >= 0.5 else 0.0
            else:
                hard = _round_half_even(alpha[o][j])
            searched[o][j] = (q[o][j] + hard) * scale
    for o in range(rows):
        for t in range(len(xl)):
            acc = 0.0
            for j in range(cols):
                acc += xl[t][j] * (searched[o][j] - wl[o][j])
            mse += acc * acc
    mse /= cols

    return {
        "q": torch.tensor(q, dtype=torch.int64),
        "r": torch.tensor(r, dtype=dtype),
        "h": torch.tensor(h, dtype=dtype),
        "alpha": torch.tensor(alpha, dtype=dtype),
        "betas": torch.tensor([(1.0 + (i + 1) / total) / log_lam for i in range(total)],
                              dtype=dtype),
        "soft_loss": torch.tensor(soft, dtype=dtype),
        "searched_weight": torch.tensor(searched, dtype=dtype),
        "chosen_offset": torch.tensor(chosen, dtype=dtype),
        "searched_mse": torch.tensor(mse, dtype=dtype),
    }


def _case(rows=2, cols=3, cal=3, num_iters=10, steps=3, scale=1.0, lam=5.0,
          dtype=torch.float64, seed=11):
    generator = torch.Generator().manual_seed(seed)
    W = torch.randn((rows, cols), generator=generator, dtype=torch.float32)
    X = torch.randn((cal, cols), generator=generator, dtype=torch.float32)
    return (W.to(dtype), X.to(dtype), scale, num_iters, steps, lam)


def _example():
    """公开示例一：X 是单位阵，误差退化成本元素平方误差之和除以列数。"""
    W = torch.tensor([[0.34, 0.90, 1.55], [0.99998, -0.72, 2.0]], dtype=torch.float64)
    X = torch.eye(3, dtype=torch.float64)
    return (W, X, 1.0, 10, 3, 5.0)


def _tie_example():
    """公开示例二：q == 0 覆盖把 h 钉在 1，候选 0 / +1/1024 / +1 完全并列。"""
    W = torch.tensor([[0.99998], [0.05]], dtype=torch.float64)
    X = torch.tensor([[20.0], [-10.0]], dtype=torch.float64)
    return (W, X, 1.0, 10, 3, 5.0)


def _close(a, b):
    tol = 3e-5 if a.dtype == torch.float32 else 2e-9
    torch.testing.assert_close(a, b, rtol=tol, atol=tol)


def _verify(solve, args):
    """结构 + 数值双重校验：与独立 oracle 一致，并检查形状/dtype/网格不变量。"""
    W, X, scale, num_iters, steps, lam = args
    before_w, before_x = W.clone(), X.clone()
    got = solve(*args)
    want = _oracle(*args)
    assert isinstance(got, dict) and set(got) == set(KEYS), f"键不对：{sorted(got)}"
    assert torch.equal(W, before_w) and torch.equal(X, before_x), "输入被就地修改了"
    rows, cols = W.shape
    for key in ("q", "r", "h", "alpha", "searched_weight", "chosen_offset"):
        assert isinstance(got[key], torch.Tensor), f"{key} 不是张量"
        assert got[key].shape == (rows, cols), f"{key} 形状 {tuple(got[key].shape)}"
        assert got[key].device == W.device, f"{key} 设备不符"
        assert torch.isfinite(got[key]).all(), f"{key} 出现非有限值"
    assert got["q"].dtype == torch.int64
    for key in ("r", "h", "alpha", "searched_weight", "chosen_offset", "betas", "soft_loss"):
        assert got[key].dtype == W.dtype, f"{key} dtype {got[key].dtype} != {W.dtype}"
    assert got["betas"].shape == (int(steps),) and got["soft_loss"].shape == (int(steps),)
    assert got["searched_mse"].dim() == 0 and got["searched_mse"].dtype == W.dtype
    assert torch.equal(got["q"], want["q"]), f"q 不符：{got['q'].tolist()} vs {want['q'].tolist()}"
    for key in ("r", "h", "alpha", "betas", "soft_loss", "searched_weight", "chosen_offset"):
        _close(got[key], want[key])
    _close(got["searched_mse"], want["searched_mse"])
    # 不变量
    _close(got["r"] + got["q"].to(W.dtype) * scale, W)
    assert (got["h"] >= 0.0).all() and (got["h"] <= 1.0).all()
    assert (got["alpha"] > 0.0).all() and (got["alpha"] < 1.0).all()
    res = float(2 ** int(num_iters))
    allowed = torch.tensor([0.0, 1.0 / res, -1.0 / res, 1.0, -1.0], dtype=torch.float64)
    flat = got["chosen_offset"].to(torch.float64).reshape(-1, 1)
    assert (flat - allowed).abs().min(dim=1).values.max() < 1e-9, "候选偏移不在集合里"
    lower = got["q"].to(W.dtype) * scale
    upper = lower + scale
    off_grid = ((got["searched_weight"] - lower).abs() > 1e-5 * max(1.0, abs(scale))) & \
               ((got["searched_weight"] - upper).abs() > 1e-5 * max(1.0, abs(scale)))
    assert not bool(off_grid.any()), "量化权重必须落在 q*scale 或 (q+1)*scale 上"
    recomputed = (X @ (got["searched_weight"] - W).T).pow(2).sum() / cols
    _close(got["searched_mse"], recomputed.to(W.dtype))
    return got


def check_example(solve):
    got = _verify(solve, _example())
    assert got["q"].tolist() == [[0, 0, 1], [0, -1, 2]], f"got {got['q'].tolist()}"
    _close(got["r"], torch.tensor([[0.34, 0.9, 0.55], [0.99998, 0.28, 0.0]],
                                 dtype=torch.float64))
    _close(got["h"], torch.tensor([[0.0, 0.9990234375, 0.5406901041666666],
                                   [0.9990234375, 0.0, 0.0]], dtype=torch.float64))
    _close(got["alpha"], torch.tensor([[0.031007766961131163, 0.9990239139880827,
                                        0.9719682639927105],
                                       [0.9990239139880827, 0.031007766961131163,
                                        0.031007766961131163]], dtype=torch.float64))
    _close(got["betas"], torch.tensor([0.8284465794128157, 1.0355582242660197,
                                       1.2426698691192237], dtype=torch.float64))
    _close(got["soft_loss"], torch.tensor([0.42640969699252196, 0.4197696698015391,
                                          0.4013846755838774], dtype=torch.float64))
    assert got["searched_weight"].tolist() == [[0.0, 1.0, 2.0], [1.0, -1.0, 2.0]]
    _close(got["chosen_offset"], torch.tensor(
        [[-1.0, -0.0009765625, -0.0009765625],
         [-0.0009765625, -1.0, -1.0]], dtype=torch.float64))
    _close(got["searched_mse"], torch.tensor(0.1355, dtype=torch.float64))


check_example.example = True


def check_example_tie_and_override(solve):
    got = _verify(solve, _tie_example())
    assert got["q"].tolist() == [[0], [0]]
    _close(got["h"], torch.tensor([[1.0], [0.0]], dtype=torch.float64))
    _close(got["alpha"], torch.tensor([[0.9999687509609075], [0.031007766961131163]],
                                      dtype=torch.float64))
    assert got["chosen_offset"].tolist() == [[0.0], [-1.0]], "并列必须取最早的候选 0"
    assert got["searched_weight"].tolist() == [[1.0], [0.0]]
    _close(got["soft_loss"], torch.tensor([0.49565801177088487, 0.4900099732982134,
                                           0.4723317661974581], dtype=torch.float64))
    _close(got["searched_mse"], torch.tensor(1.2500002, dtype=torch.float64))


check_example_tie_and_override.example = True


def check_on_grid_element(solve):
    """退化 (a)：r == 0 正好在格点上，0 与 +1 都会进位，只有 -1 能落回原格点。"""
    got = _verify(solve, (torch.tensor([[2.0]], dtype=torch.float64),
                          torch.tensor([[1.0]], dtype=torch.float64), 1.0, 10, 3, 5.0))
    assert got["q"].tolist() == [[2]] and float(got["r"]) == 0.0
    assert float(got["chosen_offset"]) == -1.0, "在格点上应提交 -1，把 h 打到 0"
    assert float(got["h"]) == 0.0
    assert float(got["searched_weight"]) == 2.0, "收尾必须回到 q + 0，即原权重本身"
    assert float(got["searched_mse"]) == 0.0
    # scale != 1 时同样落回格点
    got2 = _verify(solve, (torch.tensor([[-3.0, 0.5]], dtype=torch.float64),
                           torch.tensor([[1.0, 2.0], [0.0, 1.0]], dtype=torch.float64),
                           0.5, 10, 2, 5.0))
    assert got2["q"].tolist() == [[-6, 1]] and got2["r"].tolist() == [[0.0, 0.0]]
    assert got2["searched_weight"].tolist() == [[-3.0, 0.5]], "格点权重量化后应原样回来"
    assert float(got2["searched_mse"]) == 0.0


def check_zero_grid_override(solve):
    """退化 (b)：q == 0 -> h = 1 覆盖，即使 z = 0.5 会让插值落在判定带里。"""
    got = _verify(solve, (torch.tensor([[0.5]], dtype=torch.float64),
                          torch.tensor([[1.0]], dtype=torch.float64), 1.0, 10, 1, 5.0))
    assert int(got["q"]) == 0 and float(got["r"]) == 0.5
    # 覆盖生效：起点是 h = 1（而不是 (0.5+0.1)/1.2 = 0.5），候选 -1 把它拉到 0
    assert float(got["chosen_offset"]) == -1.0 and float(got["h"]) == 0.0
    assert abs(float(got["alpha"]) - 0.031007766961131163) < 2e-9
    assert float(got["searched_weight"]) == 0.0 and float(got["searched_mse"]) == 0.25
    # 覆盖后留在 h = 1 的情形（配合候选并列）
    got2 = _verify(solve, _tie_example())
    assert float(got2["h"][0][0]) == 1.0, "q == 0 覆盖应把 h 钉在 1"


def check_candidate_tie_keeps_earlier(solve):
    """退化 (c)：夹到 [0,1] 后候选 0 / +1/1024 / +1 的 trial h 相同 -> 目标逐位并列，取 0。"""
    args = _tie_example()
    got = _verify(solve, args)
    assert float(got["chosen_offset"][0][0]) == 0.0
    assert float(got["h"][0][0]) == 1.0
    # 同一个 h = 1 起点、更强重构偏好时依然只能选 0
    got2 = _verify(solve, (torch.tensor([[0.99998], [0.99990]], dtype=torch.float64),
                           torch.tensor([[50.0], [-30.0]], dtype=torch.float64),
                           1.0, 8, 2, 5.0))
    assert got2["chosen_offset"].tolist() == [[0.0], [0.0]], f"got {got2['chosen_offset'].tolist()}"
    assert got2["searched_weight"].tolist() == [[1.0], [1.0]]


def check_hard_decode_mask(solve):
    """退化 (d)：steps 用尽时 h 仍带内 -> 收尾用阶梯值 _h >= 0.5 进位。"""
    got = _verify(solve, (torch.tensor([[1.5012]], dtype=torch.float64),
                          torch.tensor([[1.0]], dtype=torch.float64), 1.0, 10, 1, 5.0))
    h = float(got["h"])
    assert abs(h - 0.5000234375) < 2e-9, f"got {h}"
    assert abs(h - 0.5) <= 0.5 / 1024.0 + 1e-6, "该元素应命中收尾 mask"
    assert float(got["chosen_offset"]) == -1.0 / 1024.0
    assert abs(float(got["alpha"]) - 0.9696969696969697) < 1e-9
    assert float(got["searched_weight"]) == 2.0, "mask 分支按 _h >= 0.5 进位"
    assert abs(float(got["searched_mse"]) - 0.24880144) < 1e-6


def check_identity_activations(solve):
    """退化 (e)：X = 单位阵 -> 重构项是逐元素平方误差之和除以列数，可手算。"""
    W = torch.tensor([[0.0, 0.34, 0.9], [0.99998, -0.72, 1.0]], dtype=torch.float64)
    got = _verify(solve, (W, torch.eye(3, dtype=torch.float64), 1.0, 10, 2, 5.0))
    lower, upper = got["q"].to(torch.float64), got["q"].to(torch.float64) + 1.0
    best = torch.minimum((lower - W).pow(2), (upper - W).pow(2))
    worst = torch.maximum((lower - W).pow(2), (upper - W).pow(2))
    mse = float(got["searched_mse"])
    assert float(got["chosen_offset"][0][0]) == -1.0, "w = 0 提交 -1 后 h = 0，量化回 q = 0"
    assert float(got["searched_weight"][0][0]) == 0.0 and mse < float(worst.sum() / 3.0)
    # 单位阵 + 分辨率 1024：搜索取到逐元素最近格点（RTN）的最优误差
    assert abs(mse - float(best.sum() / 3.0)) < 1e-12, f"got {mse} vs {float(best.sum()/3)}"
    assert got["searched_weight"].tolist() == [[0.0, 0.0, 1.0], [1.0, -1.0, 1.0]]


def check_multi_round_band_research(solve):
    """带内元素会被多轮重复搜索，偏移累积，chosen_offset 记最后一次提交。"""
    args = (torch.tensor([[1.5012], [0.34]], dtype=torch.float64),
            torch.tensor([[1.0], [3.0]], dtype=torch.float64), 1.0, 10, 4, 5.0)
    got = _verify(solve, args)
    assert got["soft_loss"].shape == (4,)
    assert float(got["chosen_offset"][0][0]) == -1.0 / 1024.0
    assert abs(float(got["h"][0][0]) - 0.499046875) < 1e-6, f"got {float(got['h'][0][0])}"
    assert float(got["h"][0][0]) < 0.5, "第三轮已把该元素带出判定带"
    assert float(got["chosen_offset"][1][0]) == -1.0, "第二个元素第一轮就冻结"
    assert got["betas"][0] < got["betas"][1] < got["betas"][2] < got["betas"][3]


def check_dtype_sweep(solve):
    """float32 与 float64 的同一份数值必须一致（容差内），且返回 dtype 跟随输入。"""
    W64, X64 = _case(seed=23, dtype=torch.float64)[:2]
    for args in (_case(seed=23, dtype=torch.float64), _example(), _tie_example()):
        W, X, scale, num_iters, steps, lam = args
        if W.dtype == torch.float32:
            continue
        high = solve(*args)
        low = solve(W.to(torch.float32), X.to(torch.float32), scale, num_iters, steps, lam)
        assert low["q"].dtype == torch.int64
        for key in ("r", "h", "alpha", "betas", "soft_loss", "searched_weight",
                    "chosen_offset", "searched_mse"):
            assert low[key].dtype == torch.float32, f"{key} dtype {low[key].dtype}"
            assert high[key].dtype == torch.float64
            torch.testing.assert_close(low[key].to(torch.float64), high[key],
                                       rtol=3e-5, atol=3e-5)
        assert torch.equal(low["q"], high["q"])
        assert W64.dtype == torch.float64


def check_non_contiguous_input(solve):
    """非连续（转置 / 跨步视图）输入与连续副本一致。"""
    base = torch.randn(3, 6, generator=torch.Generator().manual_seed(5),
                       dtype=torch.float32) * 1.5
    W_nc = base[:, ::2].clone().contiguous()  # 参考：连续
    W_strided = base[:, ::2]                  # 同值但非连续
    assert not W_strided.is_contiguous()
    X = torch.randn(4, 3, generator=torch.Generator().manual_seed(6),
                    dtype=torch.float32)
    args = (W_strided, X, 0.5, 6, 2, 5.0)
    got = _verify(solve, args)
    same = solve(W_nc.contiguous(), X, 0.5, 6, 2, 5.0)
    for key in KEYS:
        if key == "q":
            assert torch.equal(got[key], same[key])
        else:
            assert torch.equal(got[key], same[key]), key
    wide = torch.randn(4, 9, generator=torch.Generator().manual_seed(7),
                       dtype=torch.float32)
    transposed = wide[:3, :].T  # (3, 4) 非连续
    assert not transposed.is_contiguous()
    got_t = _verify(solve, (transposed, torch.randn(2, 3, dtype=torch.float32,
                                                   generator=torch.Generator().manual_seed(8)),
                             1.0, 4, 2, 5.0))
    assert got_t["q"].shape == (9, 3)


def check_inputs_not_mutated(solve):
    """显式断言：W / X 不被就地修改，且返回的张量不是输入的别名。"""
    W, X, args_ = _case(seed=31, dtype=torch.float32)[0], _case(seed=31, dtype=torch.float32)[1], None
    W, X = W.clone(), X.clone()
    snapshot_w, snapshot_x = W.clone(), X.clone()
    out = solve(W, X, 1.0, 6, 3, 5.0)
    assert torch.equal(W, snapshot_w) and torch.equal(X, snapshot_x)
    assert out["r"].data_ptr() != W.data_ptr()
    assert out["searched_weight"].data_ptr() != W.data_ptr()
    _verify(solve, _example())


def check_value_errors(solve):
    W = torch.ones(2, 3, dtype=torch.float32)
    X = torch.ones(2, 3, dtype=torch.float32)
    bad = [
        (W, X, 1.0, 0, 3, 5.0),        # num_iters < 1
        (W, X, 1.0, -2, 3, 5.0),
        (W, X, 0.0, 10, 3, 5.0),       # scale == 0
        (W, X, -1.0, 10, 3, 5.0),
        (W, X, 1.0, 10, 0, 5.0),       # steps < 1
        (W, X, 1.0, 10, 3, 0.0),       # lam == 0
        (W, X, 1.0, 10, 3, -2.0),      # lam < 0
        (W, X, 1.0, 10, 3, 1.0),       # lam == 1 -> log(lam) = 0
        (W, torch.ones(2, 2, dtype=torch.float32), 1.0, 10, 3, 5.0),   # 列数不匹配
        (W.unsqueeze(0), X, 1.0, 10, 3, 5.0),                          # W 不是二维
        (W, torch.ones(6, dtype=torch.float32), 1.0, 10, 3, 5.0),       # X 不是二维
        (W.double(), X, 1.0, 10, 3, 5.0),                              # dtype 不一致
    ]
    for args in bad:
        try:
            solve(*args)
        except ValueError:
            continue
        except Exception as exc:  # noqa: BLE001 - 明确拒绝其它异常类型
            raise AssertionError(f"应抛 ValueError，实抛 {type(exc).__name__}: {args[2:]}")
        raise AssertionError(f"非法输入未抛 ValueError: {args[2:]}")


def check_output_invariants(solve):
    """形状/dtype/键集合 + 量化权重的网格不变量与 searched_mse 的定义式。"""
    for seed in (101, 202, 303):
        args = _case(rows=3, cols=4, cal=2, num_iters=4, steps=2, scale=0.25,
                     lam=1000.0, seed=seed)
        got = solve(*args)
        assert set(got) == set(KEYS), f"键集合 {sorted(got)}"
        W, X, scale, num_iters, steps, lam = args
        assert got["q"].dtype == torch.int64 and got["q"].shape == W.shape
        assert got["betas"].shape == (steps,) and got["soft_loss"].shape == (steps,)
        assert got["searched_mse"].dim() == 0 and got["searched_mse"].dtype == W.dtype
        assert got["searched_weight"].dtype == W.dtype
        lower = got["q"].to(W.dtype) * scale
        assert (((got["searched_weight"] - lower).abs() < 1e-6) |
                ((got["searched_weight"] - lower - scale).abs() < 1e-6)).all()
        assert (got["r"].abs() < abs(scale) + 1e-9).all()
        assert (got["alpha"] > 0).all() and (got["alpha"] < 1).all()
        assert (got["soft_loss"].to(torch.float64) > 0).all()
        _verify(solve, args)


def check_small_num_iters_and_lam_below_one(solve):
    """num_iters = 1（分辨率 2，候选步 0.5）与 lam < 1（正则权重变负）。"""
    args = (torch.tensor([[0.4, -0.3], [1.1, 0.0]], dtype=torch.float64),
            torch.tensor([[1.0, 0.5], [0.0, 2.0]], dtype=torch.float64), 0.5, 1, 2, 0.3)
    got = _verify(solve, args)
    assert bool((got["betas"] < 0).all()), "lam < 1 时 log(lam) < 0，beta 为负"
    assert abs(float(got["betas"][1]) - 2.0 / math.log(0.3)) < 1e-9
    assert abs(float(got["betas"][0]) - 1.5 / math.log(0.3)) < 1e-9
    res = 2.0 ** 1
    assert set(torch.unique(got["chosen_offset"]).tolist()) <= {0.0, 1.0 / res, -1.0 / res, 1.0, -1.0}


def check_annealing_schedule(solve):
    """betas / soft_loss 的长度与 p_i 调度；steps = 1 时 p = 2。"""
    W, X = _example()[:2]
    for steps in (1, 2, 3, 5):
        out = solve(W, X, 1.0, 10, steps, 5.0)
        assert out["betas"].shape == (steps,) and out["soft_loss"].shape == (steps,)
        last_p = 1.0 + steps / steps
        assert abs(float(out["betas"][-1]) - last_p / math.log(5.0)) < 1e-9
        expect = [(1.0 + (i + 1) / steps) / math.log(5.0) for i in range(steps)]
        assert all(abs(float(a) - b) < 1e-9 for a, b in zip(out["betas"].tolist(), expect))
        assert bool((out["betas"][1:] > out["betas"][:-1]).all())


def check_random_against_oracle(solve):
    """120 组随机实例上与独立纯 Python oracle 九个键一致。"""
    generator = torch.Generator().manual_seed(20261031)
    lams = (0.3, 1.05, 5.0, 1000.0)
    scales = (0.25, 1.0, 3.0)
    for case in range(120):
        rows = int(torch.randint(1, 4, (1,), generator=generator))
        cols = int(torch.randint(1, 4, (1,), generator=generator))
        cal = int(torch.randint(1, 4, (1,), generator=generator))
        num_iters = int(torch.randint(1, 7, (1,), generator=generator))
        steps = int(torch.randint(1, 5, (1,), generator=generator))
        scale = scales[int(torch.randint(0, len(scales), (1,), generator=generator))]
        lam = lams[int(torch.randint(0, len(lams), (1,), generator=generator))]
        W = torch.randn(rows, cols, generator=generator, dtype=torch.float32) * 1.7
        X = torch.randn(cal, cols, generator=generator, dtype=torch.float32)
        if cols > 1 and case % 3 == 0:
            X[:, int(torch.randint(0, cols, (1,), generator=generator))] *= 6.0
        if case % 4 == 0:
            W[int(torch.randint(0, rows, (1,), generator=generator)),
              int(torch.randint(0, cols, (1,), generator=generator))] = 0.0
        if case % 5 == 0:
            W = (torch.round(W / scale) * scale)  # 一批正好落在格点上的权重
        args = (W, X, scale, num_iters, steps, lam)
        got = solve(*args)
        want = _oracle(*args)
        tag = (case, rows, cols, cal, num_iters, steps, scale, lam)
        assert torch.equal(got["q"], want["q"]), f"{tag}: q 不符"
        assert torch.equal(got["searched_weight"], want["searched_weight"]), f"{tag}: 权重不符"
        assert torch.equal(got["chosen_offset"], want["chosen_offset"]), f"{tag}: 候选不符"
        for key in ("r", "h", "alpha", "betas", "soft_loss"):
            _close(got[key], want[key])
        _close(got["searched_mse"], want["searched_mse"])
