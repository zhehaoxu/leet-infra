import math

import torch

TWO_PI = 2.0 / math.pi
FOUR_PI = 4.0 / math.pi


def _lse(row):
    top = max(row)
    return top + math.log(sum(math.exp(v - top) for v in row))


def _weighting_oracle(scheme, sigmas):
    if scheme == "sigma_sqrt":
        return [s**-2.0 for s in sigmas]
    if scheme == "cosmap":
        return [2.0 / (math.pi * (1.0 - 2.0 * s + 2.0 * s * s)) for s in sigmas]
    return [1.0] * len(sigmas)


def _argmax(row):
    best = 0
    for i in range(1, len(row)):
        if row[i] > row[best]:
            best = i
    return best


def _loss_oracle(logits, labels, lambda_conf, temperature, weights, ignore_index):
    """独立 oracle：CE、熵、两个分母、clamp_min(1) 全部用 python 标量重算。"""
    vocab = len(logits[0])
    if weights is None:
        weights = [1.0] * len(labels)

    nll, ent, correct = [], [], []
    for row, label in zip(logits, labels):
        if label == ignore_index:
            nll.append(0.0)
            ent.append(0.0)
            correct.append(False)
            continue
        nll.append(_lse(row) - row[label])
        scaled = [v / temperature for v in row]
        top = max(scaled)
        z = sum(math.exp(v - top) for v in scaled)
        probs = [math.exp(v - top) / z for v in scaled]
        ent.append(-sum(p * math.log(p) for p in probs))
        correct.append(_argmax(row) == label)

    denom_sft = max(sum(w for w, l in zip(weights, labels) if l != ignore_index), 1.0)
    sft = sum(n * w for n, w, l in zip(nll, weights, labels) if l != ignore_index) / denom_sft
    if lambda_conf == 0.0:
        return sft, sft, 0.0
    denom_conf = max(sum(w for w, c in zip(weights, correct) if c), 1.0)
    conf = sum(h * w for h, w, c in zip(ent, weights, correct) if c) / denom_conf
    return sft + lambda_conf * conf, sft, conf


def _rows(tensor):
    flat = tensor.to(torch.float64).reshape(-1, tensor.shape[-1])
    return [[float(v) for v in row] for row in flat.tolist()]


def check_example(solve):
    sigmas = torch.tensor([0.1, 0.5, 1.0], dtype=torch.float32)
    logits0 = torch.zeros(2, 3)
    labels0 = torch.zeros(2, dtype=torch.int64)

    sqrt_w, _, _, _ = solve(sigmas, "sigma_sqrt", logits0, labels0)
    assert sqrt_w.dtype == torch.float32
    assert [round(float(v), 6) for v in sqrt_w.tolist()] == [99.999992, 4.0, 1.0], sqrt_w.tolist()

    cos_w, _, _, _ = solve(sigmas, "cosmap", logits0, labels0)
    assert [round(float(v), 6) for v in cos_w.tolist()] == [0.776366, 1.273239, 0.636620], cos_w.tolist()

    ones_w, _, _, _ = solve(sigmas, "none", logits0, labels0)
    assert ones_w.tolist() == [1.0, 1.0, 1.0]

    logits = torch.tensor([[1.0, 3.0, 0.0], [0.0, 1.0, 2.0], [2.0, 0.0, 1.0]], dtype=torch.float32)
    labels = torch.tensor([1, -100, 1], dtype=torch.int64)
    _, loss, sft, conf = solve(sigmas, "sigma_sqrt", logits, labels, lambda_conf=0.5, temperature=2.0)
    assert round(float(loss), 6) == 1.741706, float(loss)
    assert round(float(sft), 6) == 1.288726, float(sft)
    assert round(float(conf), 6) == 0.905959, float(conf)

    _, loss0, sft0, conf0 = solve(sigmas, "sigma_sqrt", logits, labels)
    assert float(loss0) == float(sft0) and float(conf0) == 0.0


check_example.example = True


def check_weighting_dtype(solve):
    logits0 = torch.zeros(1, 2)
    labels0 = torch.zeros(1, dtype=torch.int64)
    sig64 = torch.tensor([0.25, 0.5], dtype=torch.float64)
    sig32 = sig64.to(torch.float32)

    sqrt64, _, _, _ = solve(sig64, "sigma_sqrt", logits0, labels0)
    assert sqrt64.dtype == torch.float32, "sigma_sqrt 在上游有 .float()，float64 输入也要降成 float32"
    assert sqrt64.tolist() == [16.0, 4.0]

    cos64, _, _, _ = solve(sig64, "cosmap", logits0, labels0)
    assert cos64.dtype == torch.float64, "cosmap 不过 .float()，dtype 跟着输入走"

    ones64, _, _, _ = solve(sig64, "unknown", logits0, labels0)
    assert ones64.dtype == torch.float64 and ones64.tolist() == [1.0, 1.0]

    cos32, _, _, _ = solve(sig32, "cosmap", logits0, labels0)
    assert cos32.dtype == torch.float32

    # 名册匹配是严格相等：大小写不同或拼错都落到 ones 分支，且不报错
    for scheme in ("none", "", "cos_map", "SIGMA_SQRT", "sigma-sqrt"):
        got, _, _, _ = solve(sig32, scheme, logits0, labels0)
        assert got.tolist() == [1.0, 1.0], scheme

    one, _, _, _ = solve(torch.tensor([1.0], dtype=torch.float32), "sigma_sqrt", logits0, labels0)
    assert one.tolist() == [1.0]


def check_cosmap_shape(solve):
    logits0 = torch.zeros(1, 2)
    labels0 = torch.zeros(1, dtype=torch.int64)
    grid = [i / 20.0 for i in range(1, 20)]
    got, _, _, _ = solve(torch.tensor(grid, dtype=torch.float64), "cosmap", logits0, labels0)
    want = _weighting_oracle("cosmap", grid)
    assert torch.allclose(got, torch.tensor(want, dtype=torch.float64), atol=1e-12)

    # bot = 2(σ-1/2)^2 + 1/2 关于 1/2 对称：权重在 σ 与 1-σ 处相等，中点 1/2 取最大值 4/π
    sym = torch.tensor([0.1, 0.9, 0.25, 0.75], dtype=torch.float64)
    got, _, _, _ = solve(sym, "cosmap", logits0, labels0)
    assert abs(float(got[0]) - float(got[1])) < 1e-12
    assert abs(float(got[2]) - float(got[3])) < 1e-12
    assert float(got[0]) < FOUR_PI, "远离 σ=1/2 时权重应小于 4/π"
    assert abs(float(got[0]) - 0.776366) < 1e-5

    mid, _, _, _ = solve(torch.tensor([0.5], dtype=torch.float64), "cosmap", logits0, labels0)
    assert abs(float(mid[0]) - FOUR_PI) < 1e-12
    for edge in (0.0 + 1e-12, 1.0):
        val, _, _, _ = solve(torch.tensor([edge], dtype=torch.float64), "cosmap", logits0, labels0)
        assert abs(float(val[0]) - TWO_PI) < 1e-6, edge


def check_sft_and_ignore(solve):
    sig = torch.tensor([0.5], dtype=torch.float32)
    logits = torch.tensor([[1.0, 3.0, 0.0], [0.0, 1.0, 2.0]], dtype=torch.float32)

    # 全部被忽略：两个损失都恰为 0，不出现 nan
    _, loss, sft, conf = solve(sig, "none", logits, torch.tensor([-100, -100]), lambda_conf=1.0)
    assert float(loss) == 0.0 and float(sft) == 0.0 and float(conf) == 0.0

    # ignore_index 是普通类号时也要整类排除：0 号 token 既正确又不参与
    all_zero = torch.zeros(2, dtype=torch.int64)
    _, loss, sft, conf = solve(sig, "none", logits, all_zero, lambda_conf=1.0, ignore_index=0)
    assert float(loss) == 0.0 and float(sft) == 0.0 and float(conf) == 0.0

    _, loss2, sft2, conf2 = solve(sig, "none", logits, all_zero, lambda_conf=1.0)
    assert float(sft2) > 0.0
    nll = [_lse([1.0, 3.0, 0.0]) - 1.0, _lse([0.0, 1.0, 2.0]) - 0.0]
    assert abs(float(sft2) - (nll[0] + nll[1]) / 2.0) < 1e-6
    assert float(conf2) == 0.0, "这两行的 argmax 都不是 0 号类，没有命中位置"

    # 正对照：两行的 argmax 都是 0 号类，有效标签与命中同时成立，置信度项应非零
    hit = torch.tensor([[3.0, 1.0, 0.0], [2.0, 0.0, 1.0]], dtype=torch.float32)
    _, loss3, sft3, conf3 = solve(sig, "none", hit, all_zero, lambda_conf=1.0)
    assert float(conf3) > 0.0
    assert abs(float(loss3) - (float(sft3) + float(conf3))) < 1e-6
    # 同一个 hit，一旦把 0 号类改成 ignore_index，两个损失都归零
    _, loss4, sft4, conf4 = solve(sig, "none", hit, all_zero, lambda_conf=1.0, ignore_index=0)
    assert float(loss4) == 0.0 and float(sft4) == 0.0 and float(conf4) == 0.0


def check_confidence_only_on_correct(solve):
    sig = torch.tensor([0.5], dtype=torch.float32)
    # 第一个位置 argmax=1 且 label=1（命中）；第二个位置 argmax=1 但 label=0（未命中）
    logits = torch.tensor([[0.0, 2.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32)
    labels = torch.tensor([1, 0], dtype=torch.int64)
    _, loss, sft, conf = solve(sig, "none", logits, labels, lambda_conf=1.0)

    scaled = [0.0, 2.0, 0.0]
    top = max(scaled)
    z = sum(math.exp(v - top) for v in scaled)
    p = [math.exp(v - top) / z for v in scaled]
    want_conf = -sum(pi * math.log(pi) for pi in p)
    assert abs(float(conf) - want_conf) < 1e-5, (float(conf), want_conf)
    assert abs(float(loss) - (float(sft) + float(conf))) < 1e-6

    # 没有任何命中位置：置信度项退化为 0，损失等于纯 CE
    none_right = torch.tensor([2, 0], dtype=torch.int64)
    _, loss2, sft2, conf2 = solve(sig, "none", logits, none_right, lambda_conf=1.0)
    assert float(conf2) == 0.0
    assert float(loss2) == float(sft2)


def check_temperature_only_affects_entropy(solve):
    sig = torch.tensor([0.5], dtype=torch.float32)
    logits = torch.tensor([[3.0, 0.0, -1.0], [0.5, 1.5, 0.0]], dtype=torch.float64)
    labels = torch.tensor([0, 1], dtype=torch.int64)

    _, _, sft0, _ = solve(sig, "none", logits, labels, lambda_conf=1.0, temperature=0.5)
    confs = []
    for temperature in (0.5, 1.0, 2.0, 5.0):
        _, _, sft, conf = solve(sig, "none", logits, labels, lambda_conf=1.0, temperature=temperature)
        assert torch.equal(sft, sft0), "温度只能缩放熵项，不能影响 CE"
        confs.append(float(conf))
    assert confs[0] < confs[1] < confs[2] < confs[3], "温度越高 softmax 越平，熵应单调上升"

    _, want_sft, want_conf = _loss_oracle(_rows(logits), [0, 1], 1.0, 2.0, None, -100)
    _, _, sft, conf = solve(sig, "none", logits, labels, lambda_conf=1.0, temperature=2.0)
    assert abs(float(conf) - want_conf) < 1e-6
    assert abs(float(sft) - want_sft) < 1e-9


def check_weight_denominator(solve):
    sig = torch.tensor([0.5], dtype=torch.float32)
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    labels = torch.tensor([0, 1], dtype=torch.int64)

    # 权重全为 0.25：分母只有 0.5，被 clamp_min(1) 截到 1，结果不是加权平均
    weights = torch.full((2,), 0.25, dtype=torch.float64)
    _, loss, sft, conf = solve(sig, "none", logits, labels, lambda_conf=0.5, per_token_weights=weights)
    _, want_sft, want_conf = _loss_oracle(_rows(logits), [0, 1], 0.5, 1.0, [0.25, 0.25], -100)
    assert abs(float(sft) - want_sft) < 1e-9, (float(sft), want_sft)
    assert abs(float(conf) - want_conf) < 1e-6

    nll = _lse([1.0, 0.0]) - 1.0
    assert abs(float(sft) - 0.25 * 2 * nll) < 1e-9, "两个 0.25 权重求和被截断到 1"

    # 权重为 0 的位置既不进分子也不进分母
    zerow = torch.tensor([0.0, 3.0], dtype=torch.float64)
    _, _, sft_z, conf_z = solve(sig, "none", logits, labels, lambda_conf=1.0, per_token_weights=zerow)
    _, want_sft_z, want_conf_z = _loss_oracle(_rows(logits), [0, 1], 1.0, 1.0, [0.0, 3.0], -100)
    assert abs(float(sft_z) - want_sft_z) < 1e-9
    assert abs(float(conf_z) - want_conf_z) < 1e-6
    assert abs(float(sft_z) - nll) < 1e-9

    # 权重 dtype 跟随 logits
    f32 = solve(sig, "none", logits.to(torch.float32), labels, per_token_weights=weights)[2]
    assert f32.dtype == torch.float32


def check_against_oracle(solve):
    torch.manual_seed(7)
    for vocab, length in ((2, 3), (4, 2), (6, 5)):
        logits = torch.randn(2, length, vocab, dtype=torch.float64) * 1.5
        labels = torch.randint(0, vocab, (2, length))
        labels[0, 0] = -100
        for lambda_conf in (0.0, 0.3, 1.0):
            for temperature in (0.5, 1.0, 2.5):
                for weights in (None, torch.rand(2, length, dtype=torch.float64)):
                    _, loss, sft, conf = solve(
                        torch.tensor([0.3], dtype=torch.float64),
                        "cosmap",
                        logits,
                        labels,
                        lambda_conf=lambda_conf,
                        temperature=temperature,
                        per_token_weights=weights,
                    )
                    w = None if weights is None else weights.tolist()
                    flat = _rows(logits)
                    flat_w = None if w is None else [v for row in w for v in row]
                    flat_labels = [v for row in labels.tolist() for v in row]
                    want_loss, want_sft, want_conf = _loss_oracle(
                        flat, flat_labels, lambda_conf, temperature, flat_w, -100
                    )
                    assert abs(float(loss) - want_loss) < 1e-6, (vocab, length, lambda_conf, temperature)
                    assert abs(float(sft) - want_sft) < 1e-9
                    assert abs(float(conf) - want_conf) < 1e-6


def check_invalid(solve):
    logits = torch.zeros(2, 3)
    labels = torch.zeros(2, dtype=torch.int64)
    ok_sig = torch.tensor([0.5], dtype=torch.float32)
    for bad in (
        lambda: solve([0.5], "none", logits, labels),
        lambda: solve(torch.tensor([1], dtype=torch.int64), "none", logits, labels),
        lambda: solve(torch.tensor([], dtype=torch.float32), "none", logits, labels),
        lambda: solve(torch.tensor([0.0], dtype=torch.float32), "sigma_sqrt", logits, labels),
        lambda: solve(torch.tensor([-0.5], dtype=torch.float32), "cosmap", logits, labels),
        lambda: solve(torch.tensor([float("nan")], dtype=torch.float32), "none", logits, labels),
        lambda: solve(torch.tensor([float("inf")], dtype=torch.float32), "none", logits, labels),
        lambda: solve(ok_sig, "none", torch.zeros(3), labels),
        lambda: solve(ok_sig, "none", torch.zeros(2, 3, dtype=torch.int64), labels),
        lambda: solve(ok_sig, "none", logits, torch.zeros(2)),
        lambda: solve(ok_sig, "none", logits, torch.zeros(3, dtype=torch.int64)),
        lambda: solve(ok_sig, "none", logits, torch.tensor([0, 3], dtype=torch.int64)),
        lambda: solve(ok_sig, "none", logits, torch.tensor([0, -1], dtype=torch.int64)),
        lambda: solve(ok_sig, "none", logits, labels, temperature=0.0),
        lambda: solve(ok_sig, "none", logits, labels, temperature=-1.0),
        lambda: solve(ok_sig, "none", logits, labels, temperature="1.0"),
        lambda: solve(ok_sig, "none", logits, labels, temperature=True),
        lambda: solve(ok_sig, "none", logits, labels, lambda_conf=float("nan")),
        lambda: solve(ok_sig, "none", logits, labels, lambda_conf="0.5"),
        lambda: solve(ok_sig, "none", logits, labels, per_token_weights=torch.ones(3)),
        lambda: solve(ok_sig, "none", logits, labels, per_token_weights=torch.ones(2, dtype=torch.int64)),
        lambda: solve(ok_sig, "none", logits, labels, ignore_index=1.5),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
