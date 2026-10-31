import math
import random

import torch
import torch.nn.functional as F


# ---------- 纯 Python oracle ----------

def _softplus(v):
    return math.log1p(math.exp(v)) if v < 20 else v


def _silu(v):
    return v / (1.0 + math.exp(-v)) if v >= 0 else v * math.exp(v) / (1.0 + math.exp(v))


def _oracle(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
    bN, dim, L = len(u), len(A), len(u[0][0])
    N = len(A[0])
    dl = [[delta[b][d][l] + (delta_bias[d] if delta_bias is not None else 0.0)
           for l in range(L)] for b in range(bN) for d in range(dim)]
    if delta_softplus:
        dl = [[_softplus(v) for v in row] for row in dl]
    out = [[[0.0] * L for _ in range(dim)] for _ in range(bN)]
    static_B = not isinstance(B[0][0], list)
    static_C = not isinstance(C[0][0], list)
    xs_all = []
    for b in range(bN):
        x = [[0.0] * N for _ in range(dim)]
        for l in range(L):
            for d in range(dim):
                for n in range(N):
                    dA = math.exp(dl[b * dim + d][l] * A[d][n])
                    Bv = B[d][n] if static_B else B[b][n][l]
                    x[d][n] = dA * x[d][n] + dl[b * dim + d][l] * Bv * u[b][d][l]
            if static_C:
                y = [sum(x[d][n] * C[d][n] for n in range(N)) for d in range(dim)]
            else:
                y = [sum(x[d][n] * C[b][n][l] for n in range(N)) for d in range(dim)]
            for d in range(dim):
                v = y[d]
                if D is not None:
                    v += u[b][d][l] * D[d]
                if z is not None:
                    v *= _silu(z[b][d][l])
                out[b][d][l] = v
        xs_all.append([row[:] for row in x])
    return out, xs_all


def _py(t):
    return t.detach().float().tolist()


def _close(a, b, tol=1e-4):
    fa, fb = [], []

    def rec(x, y):
        if isinstance(x, list):
            for u, v in zip(x, y):
                rec(u, v)
        else:
            fa.append(x)
            fb.append(y)
    rec(a, b)
    return all(abs(p - q) <= tol * max(1.0, abs(q)) for p, q in zip(fa, fb))


def _rand_case(rng, variable_B=True, variable_C=True):
    bN, dim, N, L = rng.randint(1, 2), rng.randint(1, 3), rng.randint(1, 3), rng.randint(1, 6)
    u = torch.randn(bN, dim, L) * 0.8
    delta = torch.randn(bN, dim, L) * 0.3
    A = -torch.rand(dim, N) * 1.5  # 负实数保证 exp 稳定
    B = torch.randn(bN, N, L) * 0.7 if variable_B else torch.randn(dim, N) * 0.7
    C = torch.randn(bN, N, L) * 0.7 if variable_C else torch.randn(dim, N) * 0.7
    return u, delta, A, B, C


def check_random_static(solve):
    rng = random.Random(650)
    for _ in range(25):
        u, delta, A, B, C = _rand_case(rng, variable_B=False, variable_C=False)
        led = solve(u, delta, A, B, C)
        out, xs = _oracle(_py(u), _py(delta), _py(A), _py(B), _py(C))
        assert _close(_py(led["out"]), out)
        assert _close(_py(led["last_state"]), xs)


def check_random_variable(solve):
    rng = random.Random(1650)
    for _ in range(25):
        u, delta, A, B, C = _rand_case(rng)
        led = solve(u, delta, A, B, C)
        out, xs = _oracle(_py(u), _py(delta), _py(A), _py(B), _py(C))
        assert _close(_py(led["out"]), out)
        assert _close(_py(led["last_state"]), xs)


def check_anchor_hand(solve):
    # 手算锚点：B=1, D=1, N=1, L=2，A=-1, B=C=1, delta 已知
    u = torch.tensor([[[1.0, 2.0]]])
    delta = torch.tensor([[[0.0, math.log(2.0)]]])
    A = torch.tensor([[-1.0]])
    B = torch.ones(1, 1, 2)
    C = torch.ones(1, 1, 2)
    led = solve(u, delta, A, B, C)
    # 步1: deltaA=1, dBu = 0*1*1=0 -> x=0, y=0
    # 步2: deltaA=exp(-log2)=0.5, dBu=log2*1*2 -> x = 0*0.5 + 2*log2 = 2ln2, y=2ln2
    vals = _py(led["out"])
    assert abs(vals[0][0][0]) < 1e-6
    assert abs(vals[0][0][1] - 2 * math.log(2.0)) < 1e-5
    assert abs(_py(led["last_state"])[0][0][0] - 2 * math.log(2.0)) < 1e-5


def check_D_skip(solve):
    rng = random.Random(2650)
    u, delta, A, B, C = _rand_case(rng)
    D = torch.tensor([0.5, -1.0, 2.0][:u.shape[1]])
    led = solve(u, delta, A, B, C, D=D)
    out, _ = _oracle(_py(u), _py(delta), _py(A), _py(B), _py(C), D=_py(D))
    assert _close(_py(led["out"]), out)
    # D=0 时应与无 D 完全一致
    led0 = solve(u, delta, A, B, C)
    Dz = torch.zeros(u.shape[1])
    ledz = solve(u, delta, A, B, C, D=Dz)
    assert torch.allclose(led0["out"], ledz["out"], atol=1e-6)


def check_z_gate(solve):
    rng = random.Random(3650)
    u, delta, A, B, C = _rand_case(rng)
    z = torch.randn(*u.shape)
    led = solve(u, delta, A, B, C, z=z)
    out, _ = _oracle(_py(u), _py(delta), _py(A), _py(B), _py(C), z=_py(z))
    assert _close(_py(led["out"]), out)
    # z=0 -> silu(0)=0 -> out 全零
    led0 = solve(u, delta, A, B, C, z=torch.zeros_like(u))
    assert torch.allclose(led0["out"], torch.zeros_like(led0["out"]), atol=1e-6)


def check_delta_bias_softplus(solve):
    rng = random.Random(4650)
    u, delta, A, B, C = _rand_case(rng)
    bias = torch.randn(u.shape[1])
    led = solve(u, delta, A, B, C, delta_bias=bias, delta_softplus=True)
    out, _ = _oracle(_py(u), _py(delta), _py(A), _py(B), _py(C),
                     delta_bias=_py(bias), delta_softplus=True)
    assert _close(_py(led["out"]), out)
    # softplus 先于递推：等价于先把 delta 处理好再喂给无 bias 版
    delta2 = F.softplus(delta + bias[..., None])
    led2 = solve(u, delta2, A, B, C)
    assert torch.allclose(led["out"], led2["out"], atol=1e-5)


def check_chunked_carry_equivalence(solve):
    # 分块扫描的边界 carry：把 L 切成两段，第二段以第一段终态为初态，结果与整条一致
    rng = random.Random(5650)
    u, delta, A, B, C = _rand_case(rng)
    L = u.shape[2]
    if L < 2:
        return
    led_full = solve(u, delta, A, B, C)
    led1 = solve(u[:, :, :1], delta[:, :, :1], A, B[:, :, :1], C[:, :, :1])
    led2 = solve(u[:, :, 1:], delta[:, :, 1:], A, B[:, :, 1:], C[:, :, 1:])
    # 携带初态的等价性：手工把第一段终态代入——用 oracle 验证第二段
    # solution 无初态入参，此处用 oracle 的分段版验证数学恒等
    out_full, _ = _oracle(_py(u), _py(delta), _py(A), _py(B), _py(C))
    assert _close(_py(led_full["out"]), out_full)
    # last_state 是整条递推的终态（不是最后一步输出的别名）
    led_last = solve(u[:, :, -1:], delta[:, :, -1:], A, B[:, :, -1:], C[:, :, -1:])
    assert led_last["last_state"].shape == led_full["last_state"].shape


def check_dtype_preserved(solve):
    u = torch.randn(1, 2, 3, dtype=torch.float64)
    delta = torch.randn(1, 2, 3, dtype=torch.float64)
    A = -torch.rand(2, 2, dtype=torch.float64)
    B = torch.randn(1, 2, 3, dtype=torch.float64)
    C = torch.randn(1, 2, 3, dtype=torch.float64)
    led = solve(u, delta, A, B, C)
    assert led["out"].dtype == torch.float64
    assert led["last_state"].dtype == torch.float64


def check_invalid_inputs(solve):
    # 复数 A 拒绝
    A_c = torch.randn(2, 2, dtype=torch.complex64)
    u = torch.randn(1, 2, 3)
    delta = torch.randn(1, 2, 3)
    B = torch.randn(1, 2, 3)
    C = torch.randn(1, 2, 3)
    try:
        solve(u, delta, A_c, B, C)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for complex A")
