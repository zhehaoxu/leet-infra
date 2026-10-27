import math

import torch

# 示例期望值（x 手写 (1,2,4)，eps=1e-6 默认；参考实现实测回填，>=6 位小数）
EXAMPLE_X = [[1.0, 2.0, 3.0, 4.0],
             [5.0, 6.0, 7.0, 8.0]]
# powernorm：每列 S=(x0^2+x1^2)/2，y=x/sqrt(S+eps)
EXAMPLE_PN = [[0.2773500681, 0.4472135901, 0.5570859909, 0.6324555278],
              [1.3867504597, 1.3416407108, 1.2998672724, 1.2649110556]]
# attn-pool：同一 x 视作 (B=1,L=2,D=4)，q=[1,-1,0.5,0]，logits=[0.25,1.25]
EXAMPLE_Q = [1.0, -1.0, 0.5, 0.0]
EXAMPLE_ATTN = [3.9242343903, 4.9242343903, 5.9242343903, 6.9242343903]


def _det(*shape, dtype=torch.float32):
    """确定性输入（不调用任何随机数发生器）：arange 的 sin/cos 组合花纹。"""
    n = 1
    for s in shape:
        n *= s
    idx = torch.arange(n, dtype=torch.float64)
    v = torch.sin(idx * 0.37) * 3.0 + torch.cos(idx * 0.11) * 1.5
    return v.reshape(shape).to(dtype)


def _oracle_pn(x):
    """powernorm 独立循环 oracle：纯 Python 逐 (n,m) 位置重算，全程 float64。

    与解法的向量化 mean/pow 路径不同：显式求和通道平方、显式除法。
    """
    nd = x.dim()
    n, c = x.shape[0], x.shape[1]
    m_count = x[0, 0].numel()
    xf = x.reshape(n, c, m_count).to(torch.float64).tolist()
    eps = 1e-6
    out = [[[0.0] * m_count for _ in range(c)] for _ in range(n)]
    for i in range(n):
        for m in range(m_count):
            s = 0.0
            for ch in range(c):
                s += xf[i][ch][m] * xf[i][ch][m]
            denom = math.sqrt(s / c + eps)
            for ch in range(c):
                out[i][ch][m] = xf[i][ch][m] / denom
    res = torch.tensor(out, dtype=torch.float64).reshape(x.shape)
    return res.to(x.dtype)


def _oracle_attn(x, q):
    """attn-pool 独立循环 oracle：手动 softmax（减最大值）+ 显式加权和，float64。"""
    b, l, d = x.shape
    xf = x.to(torch.float64).tolist()
    qf = q.to(torch.float64).tolist()
    scale = math.sqrt(d)
    outs = []
    for bi in range(b):
        logits = []
        for li in range(l):
            acc = 0.0
            for di in range(d):
                acc += xf[bi][li][di] * qf[di]
            logits.append(acc / scale)
        mx = max(logits)
        exps = [math.exp(v - mx) for v in logits]
        tot = sum(exps)
        attn = [e / tot for e in exps]
        row = [sum(attn[li] * xf[bi][li][di] for li in range(l)) for di in range(d)]
        outs.append(row)
    return torch.tensor(outs, dtype=torch.float64).to(x.dtype)


def check_example(solve):
    x = torch.tensor(EXAMPLE_X, dtype=torch.float32).reshape(1, 2, 4)
    pn = solve(x)  # 默认 mode="powernorm", eps=1e-6
    assert pn.shape == (1, 2, 4) and pn.dtype == torch.float32, (pn.shape, pn.dtype)
    exp_pn = torch.tensor(EXAMPLE_PN, dtype=torch.float32).reshape(1, 2, 4)
    assert torch.allclose(pn, exp_pn, atol=1e-6, rtol=1e-6), (pn.tolist(), exp_pn.tolist())
    q = torch.tensor(EXAMPLE_Q, dtype=torch.float32)
    attn = solve(x, mode="attn-pool", seed=q)
    assert attn.shape == (1, 4), attn.shape
    exp_attn = torch.tensor(EXAMPLE_ATTN, dtype=torch.float32).reshape(1, 4)
    assert torch.allclose(attn, exp_attn, atol=1e-6, rtol=1e-6), attn.tolist()


check_example.example = True


def check_powernorm_oracle(solve):
    for shape in ((2, 3, 5), (1, 4, 1), (2, 2, 3, 4), (3, 1, 7)):
        x = _det(*shape)
        got = solve(x)
        exp = _oracle_pn(x)
        assert got.shape == x.shape and got.dtype == x.dtype, (shape, got.shape, got.dtype)
        assert torch.allclose(got, exp, atol=1e-6, rtol=1e-6), \
            (shape, (got - exp).abs().max().item())
    # eps 变体
    x = _det(2, 3, 5)
    got = solve(x, eps=0.01)
    exp = _oracle_pn_eps(x, 0.01)
    assert torch.allclose(got, exp, atol=1e-6, rtol=1e-6), (got - exp).abs().max().item()


def _oracle_pn_eps(x, eps):
    """eps 可调版 powernorm oracle（与 _oracle_pn 分开实现，避免共享出错面）。"""
    n, c = x.shape[0], x.shape[1]
    m_count = x[0, 0].numel()
    xf = x.reshape(n, c, m_count).to(torch.float64).tolist()
    out = [[[0.0] * m_count for _ in range(c)] for _ in range(n)]
    for i in range(n):
        for m in range(m_count):
            s = sum(xf[i][ch][m] ** 2 for ch in range(c))
            denom = math.sqrt(s / c + eps)
            for ch in range(c):
                out[i][ch][m] = xf[i][ch][m] / denom
    return torch.tensor(out, dtype=torch.float64).reshape(x.shape).to(x.dtype)


def check_powernorm_rms_property(solve):
    """RMS 性质：eps 足够小时每个 (n,m) 位置的 sqrt(mean_C(y^2)) ≈ 1；
    eps 变大后明显小于 1（eps 的实际影响）。"""
    x = torch.tensor([[[1.0, -1.5],
                       [2.0, 0.5],
                       [-3.0, 1.0]]])  # (1,3,2)，通道均方 S≈4.67 / 1.17
    y = solve(x)
    ms = y.pow(2).mean(dim=1)          # (1,2)
    rms = torch.sqrt(ms)
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5, rtol=1e-5), rms.tolist()
    # eps=0.25 时 rms = sqrt(S/(S+eps)) ≈ 0.974 / 0.907，明显偏离 1
    y2 = solve(x, eps=0.25)
    rms2 = torch.sqrt(y2.pow(2).mean(dim=1))
    assert torch.all(rms2 < 0.99) and torch.all(rms2 > 0.85), rms2.tolist()
    # rms2 与 sqrt(S/(S+eps)) 理论值对齐（oracle 重算）
    s1 = x.pow(2).mean(dim=1).to(torch.float64)
    theo = torch.sqrt(s1 / (s1 + 0.25))
    assert torch.allclose(rms2.to(torch.float64), theo, atol=1e-6, rtol=1e-6)


def check_powernorm_sign_and_constant(solve):
    # 符号保持：除以正数不改变符号
    x = _det(2, 3, 5)
    y = solve(x)
    assert torch.equal(torch.sign(y), torch.sign(x)), "powernorm 应保持逐元素符号"
    # 常数输入 c -> y ≈ c/|c| = sign(c)（|c| >> sqrt(eps)）
    for c, expect in ((3.0, 1.0), (-2.0, -1.0)):
        xc = torch.full((2, 3, 4), c)
        yc = solve(xc)
        assert torch.allclose(yc, torch.full_like(yc, expect), atol=1e-5, rtol=1e-5), \
            (c, yc[0, 0].tolist())
    # 零输入 -> 0（0/sqrt(eps)）
    y0 = solve(torch.zeros(1, 2, 3))
    assert torch.all(y0 == 0), y0.tolist()


def check_powernorm_not_centering(solve):
    """非中心化探针：非零通道均值的输入，powernorm 不减均值——
    与 BatchNorm 式 (x-mu)/sqrt(var+eps) 的结果显著不同。"""
    x = torch.tensor([[[2.0, 4.0, 6.0],
                       [2.0, 4.0, 6.0]]])  # (1,2,3)，每列通道均值 = 列值 != 0
    y = solve(x)
    # BatchNorm 式（减均值、除标准差）独立重算
    mu = x.mean(dim=1, keepdim=True)
    var = x.pow(2).mean(dim=1, keepdim=True) - mu.pow(2)
    ybn = (x - mu) / torch.sqrt(var + 1e-6)
    assert not torch.allclose(y, ybn, atol=1e-3), (y.tolist(), ybn.tolist())
    # y 的通道均值不为 0（未中心化的直接证据）
    ch_mean = y.mean(dim=1)
    assert torch.all(ch_mean.abs() > 0.1), ch_mean.tolist()


def check_attn_pool_oracle(solve):
    cases = [
        (torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0]]]), torch.tensor([1.0, -1.0, 0.5, 0.0])),
        (_det(3, 5, 2), torch.tensor([0.7, -1.3])),
        (_det(1, 4, 3), torch.tensor([-0.2, 0.9, 1.1])),
    ]
    for x, q in cases:
        got = solve(x, mode="attn-pool", seed=q)
        exp = _oracle_attn(x, q)
        assert got.shape == (x.shape[0], x.shape[2]) and got.dtype == x.dtype, \
            (got.shape, got.dtype)
        assert torch.allclose(got, exp, atol=1e-6, rtol=1e-6), \
            (x.shape, (got - exp).abs().max().item())
    # float64 一条（含 float64 的 q）
    x64 = _det(2, 3, 4, dtype=torch.float64)
    q64 = torch.tensor([0.3, -1.2, 0.8, 2.0], dtype=torch.float64)
    got = solve(x64, mode="attn-pool", seed=q64)
    exp = _oracle_attn(x64, q64)
    assert got.dtype == torch.float64
    assert torch.allclose(got, exp, atol=1e-12, rtol=1e-12), (got - exp).abs().max().item()


def check_attn_uniform_weights(solve):
    """均匀权重探针：q 与每个 x_l 的内积相等（此处构造 x[:,l,0] 全为 1、q=e_0），
    softmax 均匀 1/L，out == x.mean(dim=1)。"""
    x = torch.tensor([[[1.0, 7.0, -2.0],
                       [1.0, 0.5, 4.0],
                       [1.0, -9.0, 3.0]]])  # (1,3,3)
    q = torch.tensor([1.0, 0.0, 0.0])
    got = solve(x, mode="attn-pool", seed=q)
    exp = x.mean(dim=1)
    assert torch.allclose(got, exp, atol=1e-6, rtol=1e-6), (got.tolist(), exp.tolist())


def check_attn_l_one(solve):
    """L=1：单一 logit 的 softmax 恒为 1，out 与 x[:,0,:] 逐位相等。"""
    x = torch.tensor([[[2.0, -3.0, 0.5]],
                      [[-1.0, 4.0, 7.0]]])  # (2,1,3)
    q = torch.tensor([1.0, 0.0, -1.0])
    got = solve(x, mode="attn-pool", seed=q)
    assert got.shape == (2, 3), got.shape
    assert torch.equal(got, x[:, 0, :]), (got.tolist(), x[:, 0, :].tolist())


def check_error_cases(solve):
    x3 = _det(1, 2, 3)

    def expect_value_error(fn, desc):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"应抛 ValueError: {desc}")

    expect_value_error(lambda: solve(x3, mode="softmax"), "非法 mode")
    expect_value_error(lambda: solve(x3, mode="attn-pool"), "attn-pool 缺 seed")
    expect_value_error(lambda: solve(_det(2, 3)), "powernorm 需要 3-D/4-D")
    expect_value_error(lambda: solve(_det(1, 2, 3, 4, 5)), "powernorm 需要 3-D/4-D")
    expect_value_error(lambda: solve(_det(2, 3), mode="attn-pool", seed=torch.tensor([1.0])),
                       "attn-pool 需要 3-D")
    expect_value_error(lambda: solve(x3, mode="attn-pool", seed=torch.tensor([1.0, 2.0])),
                       "seed 长度 != D")
    expect_value_error(lambda: solve(x3, mode="attn-pool",
                                     seed=torch.tensor([[1.0, 2.0, 3.0]])), "seed 必须一维")
    expect_value_error(lambda: solve(x3.to(torch.int64)), "dtype 必须是 float32/float64")
    expect_value_error(lambda: solve(x3, eps=0.0), "eps 必须为正数")
    expect_value_error(lambda: solve(x3, eps=-1e-3), "eps 必须为正数")
