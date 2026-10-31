import math
import torch

CHOICES = ["min", "ma", "ma2", "2", "1"]
GRID = [0.00, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
KEYS = {
    "best_alpha_index",
    "best_alpha",
    "best_wsc_choice",
    "losses",
    "s",
    "shift_vec",
    "transform_T",
    "transform_T_inv",
    "norm_comp",
    "clipped_weight",
    "lwc_bound",
}
TENSOR_KEYS = (
    "losses",
    "s",
    "shift_vec",
    "transform_T",
    "transform_T_inv",
    "clipped_weight",
    "lwc_bound",
)
MULT = 1.004999995231628  # float32 下 2*sigmoid(0.01) 的读数


def _transpose(M):
    return [[row[i] for row in M] for i in range(len(M[0]))]


def _mul(A, B):
    out = [[0.0] * len(B[0]) for _ in range(len(A))]
    for i in range(len(A)):
        Ai = A[i]
        for p in range(len(B)):
            a = Ai[p]
            if a == 0.0:
                continue
            Bp = B[p]
            for j in range(len(B[0])):
                out[i][j] += a * Bp[j]
    return out


def _threshold(w_list, choice):
    """WSC 查表：w2 = min(w)（带符号）、w3 = max|w|，按源码顺序映射。"""
    w2 = min(w_list)
    w3 = max(abs(v) for v in w_list)
    if choice == "min":
        return w2
    if choice == "ma":
        return w3 / 4.0
    if choice == "ma2":
        return w3 / 2.0
    if choice == "2":
        return w2 / 8.0
    if choice == "1":
        return w3 / 8.0
    raise AssertionError(f"unknown choice {choice!r}")


def _oracle(W, Xc, alphas, wsc_choices, wsc_w, n_bits, lwc_steps):
    """独立 oracle：纯 Python 嵌套循环重跑整条流水线，与参考实现不共享控制流。"""
    Wl = [[float(v) for v in row] for row in W.tolist()]
    Xl = [[float(v) for v in row] for row in Xc.tolist()]
    wlist = [float(v) for v in wsc_w.reshape(-1).tolist()]
    n, tokens = len(Wl), len(Xl)
    qmax = 2 ** (int(n_bits) - 1) - 1
    qmin = -(2 ** (int(n_bits) - 1))

    s1 = [max(abs(v) for v in row) for row in Wl]
    s2 = [max(abs(Xl[t][j]) for t in range(tokens)) for j in range(n)]

    losses, best = [], None
    for i, raw_alpha in enumerate(alphas):
        alpha = float(raw_alpha)
        s = [alpha * s1[j] + (1.0 - alpha) * s2[j] for j in range(n)]
        shift = [0.0 if s1[j] == 0.0 else alpha * (s1[j] - s[j]) / s1[j] for j in range(n)]
        Tm = [[s[a] * ((1.0 if a == b else 0.0) - shift[b] / n) for b in range(n)] for a in range(n)]
        Ti = [[((1.0 if a == b else 0.0) + shift[b] / n) / s[a] for b in range(n)] for a in range(n)]
        Wt = _mul(_mul(Tm, Wl), Ti)
        for choice in wsc_choices:
            thr = _threshold(wlist, choice)
            Wc = [[min(thr, max(-thr, v)) for v in row] for row in Wt]
            deq = []
            for row in Wc:
                amax = max(abs(v) for v in row)
                scale = amax / qmax
                if scale == 0.0:
                    deq.append([0.0] * n)
                else:
                    deq.append([min(qmax, max(qmin, round(v / scale))) * scale for v in row])
            Y = _mul(Xl, _transpose(Wt))
            Yq = _mul(Xl, _transpose(deq))
            total = 0.0
            for r in range(tokens):
                for c in range(n):
                    d = Y[r][c] - Yq[r][c]
                    total += d * d
            loss = total / (tokens * n)
            losses.append(loss)
            if best is None or loss < best[0]:
                best = (loss, i, choice, s, shift, Tm, Ti, Wt, Wc)

    _, i, choice, s, shift, Tm, Ti, Wt, Wc = best
    prod = _mul(Tm, Ti)
    norm_comp = max(abs(prod[a][b] - (1.0 if a == b else 0.0)) for a in range(n) for b in range(n))
    return {
        "best_alpha_index": i,
        "best_alpha": float(list(alphas)[i]),
        "best_wsc_choice": choice,
        "losses": losses,
        "s": s,
        "shift_vec": shift,
        "transform_T": Tm,
        "transform_T_inv": Ti,
        "norm_comp": norm_comp,
        "clipped_weight": Wc,
        "lwc_bound": [abs(v) * (2.0 / (1.0 + math.exp(-0.01))) for v in wlist],
    }


def _case(n=4, tokens=5, alphas=None, choices=None, bits=4, seed=7, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    W = torch.randn(n, n, generator=generator, dtype=torch.float32)
    W[0] *= 2.5
    Xc = torch.randn(tokens, n, generator=generator, dtype=torch.float32)
    Xc = Xc - Xc.mean(dim=0, keepdim=True)
    target = W.abs().amax(dim=1).clamp(min=1e-3)
    Xc = Xc / Xc.abs().amax(dim=0, keepdim=True) * target.unsqueeze(0)
    return (
        W.to(dtype),
        Xc.to(dtype),
        GRID if alphas is None else list(alphas),
        CHOICES if choices is None else list(choices),
        target.to(dtype),
        bits,
        0,
    )


def _example():
    """示例 1：alpha=0.25 与 choice="ma2" 的联合最优（题面表里的数字）。"""
    W = torch.tensor(
        [
            [3.88, 1.75, 1.45, -0.85],
            [-2.00, 1.35, 1.60, 1.30],
            [0.88, 0.85, -0.35, 0.15],
            [1.75, -0.95, 1.10, -1.50],
        ]
    )
    Xc = torch.tensor(
        [
            [-0.27, 0.39, 0.96, 1.06],
            [0.12, -0.06, -0.09, 0.46],
            [-0.03, -0.11, -0.84, 0.46],
            [0.10, 0.24, 0.31, -0.29],
            [0.08, -0.46, -0.34, -1.69],
        ]
    )
    return (W, Xc, list(GRID), list(CHOICES), W.abs().amax(dim=0).clone(), 4, 0)


def _tie_example():
    """示例 2：s1 == s2，三个 alpha 的 loss 逐位相同，并列必须取先出现者。"""
    W = torch.eye(2)
    Xc = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    return (W, Xc, [0.0, 0.2, 0.4], list(CHOICES), torch.tensor([1.0, 1.0]), 4, 0)


def _verify(solve, args):
    W, Xc, alphas, choices, wsc_w, bits, steps = args
    got = solve(*args)
    want = _oracle(*args)
    assert isinstance(got, dict), f"应返回 dict，got {type(got)}"
    assert set(got) == KEYS, f"键集合不符：{sorted(got)}"
    atol = 2e-5 if W.dtype == torch.float32 else 2e-10
    assert got["best_alpha_index"] == want["best_alpha_index"], got["best_alpha_index"]
    assert got["best_wsc_choice"] == want["best_wsc_choice"], got["best_wsc_choice"]
    assert abs(got["best_alpha"] - want["best_alpha"]) < 1e-12, got["best_alpha"]
    assert isinstance(got["best_alpha_index"], int) and isinstance(got["norm_comp"], float)
    n = W.shape[0]
    shapes = {
        "losses": (len(alphas) * len(choices),),
        "s": (n,),
        "shift_vec": (n,),
        "transform_T": (n, n),
        "transform_T_inv": (n, n),
        "clipped_weight": (n, n),
        "lwc_bound": (wsc_w.numel(),),
    }
    for key in TENSOR_KEYS:
        value = got[key]
        assert value.dtype == torch.float32, f"{key} dtype={value.dtype}"
        assert tuple(value.shape) == shapes[key], f"{key} shape {tuple(value.shape)}"
        expect = torch.tensor(want[key], dtype=torch.float64)
        assert torch.allclose(value, expect.to(torch.float32), rtol=1e-4, atol=atol), f"{key} 不符"
    assert abs(got["norm_comp"] - want["norm_comp"]) < 1e-5, (got["norm_comp"], want["norm_comp"])
    assert 0 <= got["best_alpha_index"] < len(alphas)
    return got


def check_example(solve):
    W, Xc, alphas, choices, wsc_w, bits, steps = _example()
    got = _verify(solve, (W, Xc, alphas, choices, wsc_w, bits, steps))
    assert got["best_alpha_index"] == 3, got["best_alpha_index"]
    assert abs(got["best_alpha"] - 0.25) < 1e-12, got["best_alpha"]
    assert got["best_wsc_choice"] == "ma2", got["best_wsc_choice"]
    assert torch.allclose(got["s"], torch.tensor([1.1725, 0.845, 0.94, 1.705]), atol=1e-6), got["s"].tolist()
    assert torch.allclose(
        got["shift_vec"], torch.tensor([0.174452, 0.144375, -0.017045, 0.006429]), atol=1e-6
    ), got["shift_vec"].tolist()
    assert abs(got["norm_comp"] - 4.143121e-02) < 1e-7, got["norm_comp"]
    assert torch.allclose(
        got["clipped_weight"],
        torch.tensor(
            [
                [1.94, 1.94, 1.623360, -0.577780],
                [-1.437042, 1.290990, 1.319418, 0.643622],
                [0.674053, 0.849912, -0.478703, 0.080629],
                [1.94, -1.94, 1.767677, -1.505982],
            ]
        ),
        atol=1e-5,
    ), got["clipped_weight"].tolist()
    assert torch.allclose(
        got["lwc_bound"], torch.tensor([3.899400, 1.758750, 1.608000, 1.507500]), atol=1e-5
    ), got["lwc_bound"].tolist()
    row = got["losses"][15:20]
    assert torch.allclose(
        row, torch.tensor([4.17612e-02, 2.71005e-01, 2.82965e-02, 1.44052e00, 8.50101e-01]),
        rtol=1e-4, atol=1e-9
    ), row.tolist()
    assert float(got["losses"].min()) == float(row[2]), "全局最小应落在 (0.25, ma2)"


check_example.example = True


def check_tie_example(solve):
    got = _verify(solve, _tie_example())
    assert got["best_alpha_index"] == 0, got["best_alpha_index"]
    assert got["best_wsc_choice"] == "min", got["best_wsc_choice"]
    assert float(got["losses"][0]) == 0.0, got["losses"].tolist()
    assert [float(v) for v in got["losses"][5:10]] == [float(v) for v in got["losses"][10:15]]
    assert torch.equal(got["transform_T"], torch.eye(2)), got["transform_T"].tolist()
    assert torch.equal(got["transform_T_inv"], torch.eye(2)), got["transform_T_inv"].tolist()
    assert float(got["norm_comp"]) == 0.0, got["norm_comp"]
    assert torch.equal(got["s"], torch.ones(2)) and torch.equal(got["shift_vec"], torch.zeros(2))
    assert torch.allclose(got["lwc_bound"], torch.full((2,), MULT), atol=1e-7), got["lwc_bound"].tolist()


check_tie_example.example = True


def check_alpha_zero_is_activation_scale(solve):
    W, Xc, _, choices, wsc_w, bits, steps = _example()
    got = _verify(solve, (W, Xc, [0.0], choices, wsc_w, bits, steps))
    assert torch.allclose(got["s"], Xc.abs().amax(dim=0), atol=1e-6), got["s"].tolist()
    assert torch.equal(got["shift_vec"], torch.zeros(4)), got["shift_vec"].tolist()
    assert torch.allclose(got["transform_T"], torch.diag(got["s"]), atol=1e-6)
    assert torch.allclose(got["transform_T_inv"], torch.diag(1.0 / got["s"]), atol=1e-6)
    assert float(got["norm_comp"]) < 1e-12, got["norm_comp"]
    assert got["best_alpha_index"] == 0


def check_alpha_one_is_weight_scale(solve):
    W, Xc, _, choices, wsc_w, bits, steps = _example()
    got = _verify(solve, (W, Xc, [1.0], choices, wsc_w, bits, steps))
    assert torch.allclose(got["s"], W.abs().amax(dim=1), atol=1e-6), got["s"].tolist()
    assert torch.equal(got["shift_vec"], torch.zeros(4)), got["shift_vec"].tolist()
    assert torch.allclose(got["transform_T"], torch.diag(got["s"]), atol=1e-6), got["transform_T"].tolist()
    assert torch.allclose(got["transform_T_inv"], torch.diag(1.0 / got["s"]), atol=1e-6)
    assert float(got["norm_comp"]) == 0.0, got["norm_comp"]
    diag = torch.diag(got["s"])
    assert torch.equal(got["transform_T"] - diag, torch.zeros(4, 4)), got["transform_T"].tolist()
    assert torch.allclose(got["transform_T_inv"] - torch.diag(1.0 / got["s"]), torch.zeros(4, 4), atol=1e-6)


def check_wsc_threshold_table(solve):
    """逐个 choice 单独跑，核对源码顺序映射 min->w2, ma->w3/4, ma2->w3/2, 2->w2/8, 1->w3/8。"""
    W, Xc, _, _, _, bits, steps = _example()
    wsc_w = torch.tensor([3.88, 1.75, 1.60, 1.50])
    w2, w3 = min(wsc_w.tolist()), max(abs(v) for v in wsc_w.tolist())
    expect = {"min": w2, "ma": w3 / 4.0, "ma2": w3 / 2.0, "2": w2 / 8.0, "1": w3 / 8.0}
    assert list(expect) == CHOICES, "映射表必须按源码顺序"
    for choice, thr in expect.items():
        args = (W, Xc, [0.25], [choice], wsc_w, bits, steps)
        got = _verify(solve, args)
        assert got["best_wsc_choice"] == choice and got["losses"].shape == (1,)
        peak = float(got["clipped_weight"].abs().max())
        assert abs(peak - thr) < 1e-5, f"{choice}: 阈值 {thr} vs 裁剪后 absmax {peak}"
        assert float(got["clipped_weight"].abs().max()) <= thr + 1e-6


def check_lwc_init_bound_constant(solve):
    """LWC 初值界 = |wsc_w| * 2*sigmoid(0.01)，倍数逐元素钉死。"""
    W = torch.eye(3)
    Xc = torch.zeros(2, 3)
    wsc_w = torch.tensor([1.0, 2.0, 0.5])
    got = _verify(solve, (W, Xc, [0.2], ["min"], wsc_w, 4, 0))
    bound = [float(v) for v in got["lwc_bound"]]
    assert abs(bound[0] - MULT) < 1e-9, repr(bound[0])
    assert abs(bound[1] - 2.0 * MULT) < 1e-6, repr(bound[1])
    assert abs(bound[2] - 0.5 * MULT) < 1e-6, repr(bound[2])
    assert abs(bound[0] / 1.0 - 1.0049999583337499) < 1e-6
    assert abs(MULT - 1.0049958) < 5e-6, "上游引用的数字就是这个倍数的 float32 读数"
    assert abs(2.0 / (1.0 + math.exp(-0.01)) - 1.0049999583337499) < 1e-12
    assert got["lwc_bound"].shape == (3,)


def check_norm_comp_is_identity_deviation(solve):
    """norm_comp 是 ||T T_inv - I||_inf（用返回的两张矩阵复算），并软核上游 0.05 那条断言。"""
    for args in (_example(), _tie_example(), _case(seed=31), _case(seed=52, n=3, tokens=4)):
        got = solve(*args)
        prod = got["transform_T"].double() @ got["transform_T_inv"].double()
        eye = torch.eye(prod.shape[0], dtype=torch.float64)
        recomputed = float((prod - eye).abs().max())
        assert abs(recomputed - got["norm_comp"]) < 2e-5, (recomputed, got["norm_comp"])
        assert got["norm_comp"] >= 0.0 and math.isfinite(got["norm_comp"])
    assert solve(*_example())["norm_comp"] < 0.05, "示例 1 应通过上游那条 <0.05 的软检查"
    assert solve(*_tie_example())["norm_comp"] == 0.0


def check_norm_comp_probe_exceeds_soft_bound(solve):
    """故意让 s1 与 s2 量级失配：norm_comp 越过硬检查阈值，但实现只上报、不抛异常。"""
    W, Xc, _, choices, wsc_w, bits, steps = _example()
    got = _verify(solve, (W, Xc * 8.0, [0.15], choices, wsc_w, bits, steps))
    assert got["norm_comp"] >= 0.05, got["norm_comp"]
    assert torch.isfinite(got["losses"]).all() and torch.isfinite(got["clipped_weight"]).all()
    assert torch.isfinite(got["transform_T"]).all() and torch.isfinite(got["transform_T_inv"]).all()


def check_dtype_sweep(solve):
    """float32 / float64 输入都要与 oracle 一致，且同一份数据的两种 dtype 结果逐位相同。"""
    args = _example()
    f32 = solve(*args)
    f64 = solve(args[0].double(), args[1].double(), args[2], args[3], args[4].double(), args[5], args[6])
    assert f32["best_alpha_index"] == f64["best_alpha_index"]
    assert f32["best_wsc_choice"] == f64["best_wsc_choice"]
    assert f32["norm_comp"] == f64["norm_comp"]
    for key in TENSOR_KEYS:
        assert f32[key].dtype == torch.float32 and f64[key].dtype == torch.float32, key
        assert torch.equal(f32[key], f64[key]), key
    for seed in (3, 91, 271):
        for dtype in (torch.float32, torch.float64):
            _verify(solve, _case(n=3, tokens=4, seed=seed, dtype=dtype, bits=8))


def check_noncontiguous_inputs(solve):
    W, Xc, alphas, choices, wsc_w, bits, steps = _example()
    base = solve(W, Xc, alphas, choices, wsc_w, bits, steps)
    W_nc = W.t().contiguous().t()
    Xc_nc = torch.cat([Xc, Xc], dim=1)[:, : Xc.shape[1]]
    wsc_nc = torch.cat([wsc_w, wsc_w], dim=0)[: wsc_w.numel()]
    assert not W_nc.is_contiguous() and not Xc_nc.is_contiguous()
    got = solve(W_nc, Xc_nc, alphas, choices, wsc_nc, bits, steps)
    assert got["best_wsc_choice"] == base["best_wsc_choice"]
    for key in TENSOR_KEYS:
        assert torch.equal(got[key], base[key]), key
    W_view = torch.cat([W, W], dim=1)[:, : W.shape[1]]
    assert not W_view.is_contiguous()
    got2 = solve(W_view, Xc, alphas, choices, wsc_w, bits, steps)
    for key in TENSOR_KEYS:
        assert torch.equal(got2[key], base[key]), key
    got3 = solve(W.t(), Xc, alphas, choices, wsc_w, bits, steps)
    assert not torch.equal(got3["clipped_weight"], base["clipped_weight"]), "转置是另一张权重"


def check_inputs_are_not_mutated(solve):
    for args in (_example(), _tie_example(), _case(seed=13, n=3, tokens=3)):
        copies = [a.clone() if torch.is_tensor(a) else (list(a) if isinstance(a, list) else a) for a in args]
        solve(*args)
        for before, after in zip(args, copies):
            if torch.is_tensor(after):
                assert torch.equal(before, after), "输入张量被改写了"
            elif isinstance(after, list):
                assert before == after, "输入列表被改写了"


def check_degenerate_layers(solve):
    """常数权重（w2 == w3）与全零输出行（s1 含 0）两条退化路径。"""
    W = torch.full((2, 2), 0.5)
    Xc = torch.tensor([[0.25, -0.25], [-0.25, 0.25]])
    wsc_w = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    assert float(wsc_w.min()) == float(wsc_w.abs().max()), "该例要制造 w2 == w3"
    got = _verify(solve, (W, Xc, list(GRID), list(CHOICES), wsc_w, 4, 0))
    assert float(got["losses"][0]) == 0.0, got["losses"].tolist()
    assert got["best_alpha_index"] == 0 and got["best_wsc_choice"] == "min"
    assert torch.allclose(got["lwc_bound"], torch.full((4,), 0.5 * MULT), atol=1e-7)
    assert torch.allclose(got["s"], torch.tensor([0.25, 0.25]), atol=1e-6), got["s"].tolist()
    assert torch.equal(got["shift_vec"], torch.zeros(2)), got["shift_vec"].tolist()

    W2 = torch.tensor([[0.0, 0.0], [0.75, -1.25]])
    X2 = torch.tensor([[0.6, 0.2], [-0.6, -0.2]])
    got2 = _verify(solve, (W2, X2, [0.2], list(CHOICES), torch.tensor([1.25, 0.6]), 4, 0))
    assert float(got2["shift_vec"][0]) == 0.0, got2["shift_vec"].tolist()
    assert abs(float(got2["s"][0]) - 0.8 * 0.6) < 1e-6, got2["s"].tolist()
    assert abs(float(got2["s"][1]) - (0.2 * 1.25 + 0.8 * 0.2)) < 1e-6, got2["s"].tolist()
    assert torch.isfinite(got2["losses"]).all() and got2["best_alpha_index"] == 0


def check_lwc_steps_are_inert(solve):
    args = _example()
    ref = solve(*args)
    for steps in range(1, 9):
        other = solve(args[0], args[1], args[2], args[3], args[4], args[5], steps)
        assert other["best_alpha_index"] == ref["best_alpha_index"]
        assert other["best_wsc_choice"] == ref["best_wsc_choice"]
        assert other["norm_comp"] == ref["norm_comp"]
        assert other["best_alpha"] == ref["best_alpha"]
        for key in KEYS - {
            "best_alpha_index",
            "best_alpha",
            "best_wsc_choice",
            "norm_comp",
        }:
            assert torch.equal(other[key], ref[key]), key


def check_invalid_inputs(solve):
    W, Xc, alphas, choices, wsc_w, bits, steps = _example()
    for bad in ([-0.01], [1.01], [float("nan")], ["x"]):
        try:
            solve(W, Xc, bad, choices, wsc_w, bits, steps)
        except ValueError:
            pass
        else:
            raise AssertionError(f"alpha={bad} 应抛 ValueError")
    for bad_args in (
        (W, Xc, [], choices, wsc_w, bits, steps),
        (W, Xc, alphas, [], wsc_w, bits, steps),
        (W, Xc, alphas, ["max"], wsc_w, bits, steps),
        (W, Xc[:, :3], alphas, choices, wsc_w, bits, steps),
        (W[:, :3], Xc, alphas, choices, wsc_w, bits, steps),
        (W.reshape(1, -1), Xc, alphas, choices, wsc_w, bits, steps),
        (W, Xc, alphas, choices, wsc_w[:0], bits, steps),
        (W, Xc, alphas, choices, wsc_w, bits, -1),
        (torch.zeros(2, 2), torch.zeros(2, 2), [0.0], ["min"], torch.tensor([1.0]), 4, 0),
    ):
        try:
            solve(*bad_args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"应抛 ValueError：{bad_args}")
    for bad_bits in (1, 17, 0):
        try:
            solve(W, Xc, alphas, choices, wsc_w, bad_bits, steps)
        except ValueError:
            pass
        else:
            raise AssertionError(f"n_bits={bad_bits} 应抛 ValueError")


def check_loss_flatten_order_and_ties(solve):
    """losses 摊平顺序 = alpha 外层 / choice 内层；重复 alpha 给同一行、并列取下标小者。"""
    W, Xc, alphas, choices, wsc_w, bits, steps = _example()
    got = solve(W, Xc, [0.25, 0.25, 0.0], choices, wsc_w, bits, steps)
    assert got["losses"].shape == (15,), tuple(got["losses"].shape)
    assert torch.equal(got["losses"][0:5], got["losses"][5:10]), "重复 alpha 必须给出同一行"
    assert got["best_alpha_index"] == 0, got["best_alpha_index"]
    assert got["best_wsc_choice"] == "ma2", got["best_wsc_choice"]
    full = solve(W, Xc, alphas, choices, wsc_w, bits, steps)
    flat = full["losses"]
    first = int((flat == flat.min()).nonzero(as_tuple=True)[0][0])
    assert first == full["best_alpha_index"] * len(choices) + choices.index(full["best_wsc_choice"])
    assert abs(float(flat[first]) - float(flat.min())) < 1e-12
    for i, alpha in enumerate(alphas):
        one = solve(W, Xc, [alpha], choices, wsc_w, bits, steps)["losses"]
        assert torch.allclose(flat[i * 5 : (i + 1) * 5], one, rtol=1e-4, atol=1e-9), alpha


def check_random_against_oracle(solve):
    """随机权重 / 激活 / 网格 / 位数上与独立 oracle 一致。"""
    g = torch.Generator().manual_seed(20261030)
    for _ in range(40):
        n = int(torch.randint(2, 5, (1,), generator=g))
        tokens = int(torch.randint(2, 6, (1,), generator=g))
        W = torch.randn(n, n, generator=g, dtype=torch.float32) * 1.5
        if int(torch.randint(0, 2, (1,), generator=g)) == 1:
            W[:, 0] *= 3.0
        Xc = torch.randn(tokens, n, generator=g, dtype=torch.float32)
        Xc = Xc - Xc.mean(dim=0, keepdim=True)
        if n > 2:
            Xc[:, -1] *= float(torch.randint(2, 4, (1,), generator=g))
        wsc_w = W.abs().amax(dim=1).clone()
        if int(torch.randint(0, 3, (1,), generator=g)) == 0:
            wsc_w = torch.randn(2, n, generator=g, dtype=torch.float32).abs()
        args = (
            W,
            Xc,
            [0.0, 0.05, 0.15, 0.4],
            ["ma", "min", "1", "ma2"],
            wsc_w,
            int(torch.randint(2, 9, (1,), generator=g)),
            int(torch.randint(0, 4, (1,), generator=g)),
        )
        _verify(solve, args)
