import torch

# 示例：T=1、N=1、E=2、Hid=1，四个门的 bias 全 0；门顺序 (i, f, g, o)。
# gates = x@W_ihᵀ + h0@W_hhᵀ = [0, 0, (−2−2)+4, 0] = [0,0,0,0]
# ⇒ i=g=f=o 的自变量都是 0：i=f=o=0.5、g=tanh(0)=0
# c1 = 0.5*0 + 0.5*0 = 0 ⇒ tanh(c1)=0 ⇒ h1 = 0
# dh=1, dc_next=0 ⇒ dct = 0 + 1·o·(1−0) = 0.5
# di = dct*g = 0、dg = dct*i = 0.25、df = dct*c0 = 0、do = dh·tanh(c1) = 0
# dgates = [0, 0, 0.25, 0] ⇒ dx = 0.25*W_ih[2] = [−0.5,−0.5]；dW_ih 第 2 行 = 0.25*x = [0.25,0.25]
# dW_hh 第 2 行 = 0.25*h0 = 0.25；db_ih = db_hh = [0,0,0.25,0]；dh0 = 0.25*W_hh[2] = 1；dc0 = dct*f = 0.25
EXAMPLE_XS = [[[1.0, 1.0]]]
EXAMPLE_H0 = [[1.0]]
EXAMPLE_C0 = [[0.0]]
EXAMPLE_W_IH = [[0.0, 0.0], [0.0, 0.0], [-2.0, -2.0], [0.0, 0.0]]
EXAMPLE_W_HH = [[0.0], [0.0], [4.0], [0.0]]
EXAMPLE_DHT = [[1.0]]
EXAMPLE_DCT = [[0.0]]
EXAMPLE_DXS = [[[-0.5, -0.5]]]
EXAMPLE_DW_IH = [[0.0, 0.0], [0.0, 0.0], [0.25, 0.25], [0.0, 0.0]]
EXAMPLE_DW_HH = [[0.0], [0.0], [0.25], [0.0]]
EXAMPLE_DB = [0.0, 0.0, 0.25, 0.0]
EXAMPLE_DH0 = [[1.0]]
EXAMPLE_DC0 = [[0.25]]


def _oracle(xs, h0, c0, W_ih, W_hh, b_ih, b_hh, dhT, dcT):
    """用 nn.LSTM（cuDNN/CPPU 内核 + autograd）做独立参考，顺带钉死门顺序与 bias 口径。"""
    T, N, E = xs.shape
    Hid = h0.shape[1]
    mod = torch.nn.LSTM(E, Hid).to(xs.dtype)          # 与输入的 float64 对齐
    with torch.no_grad():
        mod.weight_ih_l0.copy_(W_ih)
        mod.weight_hh_l0.copy_(W_hh)
        mod.bias_ih_l0.copy_(b_ih)
        mod.bias_hh_l0.copy_(b_hh)
    xs_ = xs.clone().requires_grad_(True)
    h_ = h0.unsqueeze(0).clone().requires_grad_(True)      # nn.LSTM 要 (layers, N, Hid)
    c_ = c0.unsqueeze(0).clone().requires_grad_(True)
    _, (hn, cn) = mod(xs_, (h_, c_))
    loss = (hn[0] * dhT).sum() + (cn[0] * dcT).sum()
    grads = torch.autograd.grad(loss, [xs_, h_, c_, mod.weight_ih_l0, mod.weight_hh_l0,
                                       mod.bias_ih_l0, mod.bias_hh_l0])
    return [grads[0], grads[1][0], grads[2][0]] + list(grads[3:])  # dxs, dh0, dc0, dW..., db...


def _random_case(seed, T, N, E, Hid, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=g, dtype=dtype)

    xs = rnd(T, N, E)
    h0 = rnd(N, Hid)
    c0 = rnd(N, Hid)
    W_ih = rnd(4 * Hid, E) * 0.4
    W_hh = rnd(4 * Hid, Hid) * 0.4
    b_ih = rnd(4 * Hid) * 0.2
    b_hh = rnd(4 * Hid) * 0.2
    dhT = rnd(N, Hid)
    dcT = rnd(N, Hid)
    return xs, h0, c0, W_ih, W_hh, b_ih, b_hh, dhT, dcT




def check_example(solve):
    xs = torch.tensor(EXAMPLE_XS)
    h0 = torch.tensor(EXAMPLE_H0)
    c0 = torch.tensor(EXAMPLE_C0)
    W_ih = torch.tensor(EXAMPLE_W_IH)
    W_hh = torch.tensor(EXAMPLE_W_HH)
    b = torch.zeros(4, dtype=torch.float64)
    dhT = torch.tensor(EXAMPLE_DHT)
    dcT = torch.tensor(EXAMPLE_DCT)
    out = solve(xs.double(), h0.double(), c0.double(), W_ih.double(), W_hh.double(), b, b,
                dhT.double(), dcT.double())
    assert isinstance(out, tuple) and len(out) == 7
    dxs, dW_ih, dW_hh, db_ih, db_hh, dh0, dc0 = out
    assert dxs.shape == (1, 1, 2) and dW_ih.shape == (4, 2) and dW_hh.shape == (4, 1)
    assert db_ih.shape == (4,) and db_hh.shape == (4,)
    assert torch.allclose(dxs, torch.tensor(EXAMPLE_DXS, dtype=torch.float64), atol=1e-9), dxs
    assert torch.allclose(dW_ih, torch.tensor(EXAMPLE_DW_IH, dtype=torch.float64), atol=1e-9), dW_ih
    assert torch.allclose(dW_hh, torch.tensor(EXAMPLE_DW_HH, dtype=torch.float64), atol=1e-9), dW_hh
    assert torch.allclose(db_ih, torch.tensor(EXAMPLE_DB, dtype=torch.float64), atol=1e-9), db_ih
    assert torch.allclose(db_hh, torch.tensor(EXAMPLE_DB, dtype=torch.float64), atol=1e-9), db_hh
    assert torch.allclose(dh0, torch.tensor(EXAMPLE_DH0, dtype=torch.float64), atol=1e-9), dh0
    assert torch.allclose(dc0, torch.tensor(EXAMPLE_DC0, dtype=torch.float64), atol=1e-9), dc0


check_example.example = True


def check_matches_nn_lstm(solve):
    for seed, (T, N, E, Hid) in [(401, (5, 3, 4, 6)), (402, (1, 2, 3, 2)), (403, (7, 1, 2, 5))]:
        case = _random_case(seed, T, N, E, Hid)
        got = solve(*case)
        ref = _oracle(*case)
        order = [ref[0], ref[3], ref[4], ref[5], ref[6], ref[1], ref[2]]
        for k, (g, r) in enumerate(zip(got, order)):
            assert g.shape == r.shape, (k, g.shape, r.shape)
            assert torch.allclose(g, r, atol=1e-8), (seed, k, (g - r).abs().max())


def _forward_states(xs, h0, c0, W_ih, W_hh, b_ih, b_hh):
    """测试内独立前向：返回每一步的 (h_t, c_t)（float64，无 autograd）。"""
    hid = h0.shape[1]
    h, c = h0.double(), c0.double()
    states = []
    for t in range(xs.shape[0]):
        gates = xs[t].double() @ W_ih.double().T + h @ W_hh.double().T + (b_ih + b_hh).double()
        i = torch.sigmoid(gates[:, 0:hid])
        f = torch.sigmoid(gates[:, hid:2 * hid])
        g = torch.tanh(gates[:, 2 * hid:3 * hid])
        o = torch.sigmoid(gates[:, 3 * hid:4 * hid])
        c = f * c + i * g
        h = o * torch.tanh(c)
        states.append((h.clone(), c.clone()))
    return states


def check_bias_pairing(solve):
    # gates 里 b_ih 与 b_hh 是两段独立的全长偏置 ⇒ 两者梯度必然相同（都是 Σ_t dgates）
    case = _random_case(404, 6, 2, 3, 4)
    got = solve(*case)
    ref = _oracle(*case)
    assert torch.allclose(got[3], got[4], atol=1e-12)          # db_ih == db_hh
    assert torch.allclose(got[3], ref[5], atol=1e-8)
    assert torch.allclose(got[4], ref[6], atol=1e-8)
    # 上游梯度全 0 ⇒ 所有梯度都是 0（没有别的来源）
    xs, h0, c0, W_ih, W_hh, b_ih, b_hh, dhT, dcT = case
    z = torch.zeros_like(dhT)
    for g in solve(xs, h0, c0, W_ih, W_hh, b_ih, b_hh, z, z):
        assert torch.count_nonzero(g) == 0


def check_time_accumulation(solve):
    # dW_ih 是 Σ_t dgates_tᵀ @ x_t：只喂最后一步得到的 dW 必须小于整段，且与整段不同
    case = _random_case(409, 5, 2, 3, 4)
    xs, h0, c0, W_ih, W_hh, b_ih, b_hh, dhT, dcT = case
    full = solve(*case)
    states = _forward_states(xs, h0, c0, W_ih, W_hh, b_ih, b_hh)
    last_h, last_c = states[-2]                                 # 第 T−1 步的输入状态
    only_last = solve(xs[-1:], last_h.to(xs.dtype), last_c.to(xs.dtype),
                      W_ih, W_hh, b_ih, b_hh, dhT, dcT)
    assert not torch.allclose(full[1], only_last[1], atol=1e-6), "dW_ih 必须累加所有时间步"
    # 整段的 db 等于逐步 dgates 之和：用最后一步的 db 反推不出整段
    assert full[3].abs().sum() > only_last[3].abs().sum()
    # 与 nn.LSTM 对拍保证「累加的正好是全部时间步、不重不漏」
    ref = _oracle(*case)
    assert torch.allclose(full[1], ref[3], atol=1e-8)


def check_zero_upstream_and_linearity(solve):
    case = _random_case(405, 4, 2, 3, 3)
    xs, h0, c0, W_ih, W_hh, b_ih, b_hh, dhT, dcT = case
    z = torch.zeros_like(dhT)
    gh = solve(xs, h0, c0, W_ih, W_hh, b_ih, b_hh, dhT, z)
    gc = solve(xs, h0, c0, W_ih, W_hh, b_ih, b_hh, z, dcT)
    full = solve(xs, h0, c0, W_ih, W_hh, b_ih, b_hh, dhT, dcT)
    # BPTT 对上游梯度是线性的：两半之和等于整体
    for a, b, c in zip(gh, gc, full):
        assert torch.allclose(a + b, c, atol=1e-9), (a - c + b).abs().max()
    # 只有 dcT 时，梯度只从 c 这条路走：与 nn.LSTM 的 autograd 一致
    ref = _oracle(xs, h0, c0, W_ih, W_hh, b_ih, b_hh, z, dcT)
    order = [ref[0], ref[3], ref[4], ref[5], ref[6], ref[1], ref[2]]
    for a, r in zip(gc, order):
        assert torch.allclose(a, r, atol=1e-8)


def check_truncated_bptt_segmentation(solve):
    # 截断 BPTT 的自洽性：把 T 步在第 k 步切开，后段回传的 dh/dc 作为前段的上游，
    # 两段的权重/输入梯度分别相加必须等于整段的结果（时间累加没有漏项、没有重复计入）
    xs, h0, c0, W_ih, W_hh, b_ih, b_hh, dhT, dcT = _random_case(406, 6, 2, 3, 4)
    T = xs.shape[0]
    k = 2
    # 前向取第 k 步的 (h_k, c_k)：测试内独立实现，不用 autograd
    x64, w_ih, w_hh = xs.double(), W_ih.double(), W_hh.double()
    bias = (b_ih + b_hh).double()
    hid = h0.shape[1]
    h, c = h0.double(), c0.double()
    for t in range(k):
        gates = x64[t] @ w_ih.T + h @ w_hh.T + bias
        i = torch.sigmoid(gates[:, 0:hid])
        f = torch.sigmoid(gates[:, hid:2 * hid])
        g = torch.tanh(gates[:, 2 * hid:3 * hid])
        o = torch.sigmoid(gates[:, 3 * hid:4 * hid])
        c = f * c + i * g
        h = o * torch.tanh(c)
    seg2 = solve(xs[k:], h.to(xs.dtype), c.to(xs.dtype), W_ih, W_hh, b_ih, b_hh, dhT, dcT)
    dh_k, dc_k = seg2[5], seg2[6]
    seg1 = solve(xs[:k], h0, c0, W_ih, W_hh, b_ih, b_hh, dh_k, dc_k)
    full = solve(xs, h0, c0, W_ih, W_hh, b_ih, b_hh, dhT, dcT)
    for idx, (a, b, r) in enumerate(zip(seg1, seg2, full)):
        if idx == 0:                                   # dxs 要按时间拼接
            assert torch.allclose(torch.cat([a, b], dim=0), r, atol=1e-8)
        elif idx in (5, 6):                            # 段的 dh0/dc0 不满足可加性
            assert a.shape == b.shape == r.shape
        else:
            assert torch.allclose(a + b, r, atol=1e-8), (idx, (a + b - r).abs().max())


def check_dtype_and_invalid(solve):
    case = _random_case(407, 3, 2, 3, 4, dtype=torch.float32)
    got = solve(*case)
    dtypes = [case[0].dtype, case[3].dtype, case[4].dtype, case[5].dtype,
              case[6].dtype, case[1].dtype, case[2].dtype]
    for g, d in zip(got, dtypes):
        assert g.dtype == d, (g.dtype, d)
    xs, h0, c0, W_ih, W_hh, b_ih, b_hh, dhT, dcT = _random_case(408, 3, 2, 3, 4)
    bad = [
        (xs[0], h0, c0, W_ih, W_hh, b_ih, b_hh, dhT, dcT),                 # xs 二维
        (xs, h0.unsqueeze(0), c0, W_ih, W_hh, b_ih, b_hh, dhT, dcT),      # h0 三维
        (xs, h0, c0[:, :1], W_ih, W_hh, b_ih, b_hh, dhT, dcT),            # c0 不同形
        (xs, h0, c0, W_ih[1:], W_hh, b_ih, b_hh, dhT, dcT),               # W_ih 行数不是 4Hid
        (xs, h0, c0, W_ih, W_hh[:, :2], b_ih, b_hh, dhT, dcT),            # W_hh 列数不对
        (xs, h0, c0, W_ih, W_hh, b_ih[1:], b_hh, dhT, dcT),               # bias 长度不对
        (xs, h0, c0, W_ih, W_hh, b_ih, b_hh, dhT[:, :1], dcT),            # dhT 形状不对
        (xs, h0, c0, W_ih, W_hh, b_ih, b_hh, dhT, dcT.reshape(-1)),       # dcT 被压平
    ]
    for args in bad:
        try:
            solve(*args)
        except ValueError:
            continue
        raise AssertionError("非法入参应抛 ValueError")
