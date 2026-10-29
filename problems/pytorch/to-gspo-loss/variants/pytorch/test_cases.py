import math

import torch

LOGPS = torch.tensor([[0.10, -0.20, 0.00], [0.30, 0.00, 0.50]], dtype=torch.float64)
MASK = torch.tensor([[True, True, True], [True, True, False]], dtype=torch.bool)
ADV = torch.tensor([1.0, -1.0], dtype=torch.float64)
EXAMPLE = {
    "loss": 0.09730907112313858,
    "ratio": [0.9672161004820059, 1.161834242728283],
}


def _oracle(logp, old, mask, adv, eps):
    n = len(logp)
    ratios = []
    total = 0.0
    for i in range(n):
        acc = 0.0
        cnt = 0
        for t in range(len(logp[i])):
            if mask[i][t]:
                acc += logp[i][t] - old[i][t]
                cnt += 1
        r = math.exp(acc / cnt)
        ratios.append(r)
        a = adv[i]
        c = min(max(r, 1.0 - eps), 1.0 + eps)
        total += min(r * a, c * a)
    return ratios, -total / n


def check_example(solve):
    out = solve(LOGPS, torch.zeros_like(LOGPS), MASK, ADV, 3e-4)
    assert set(out) == {"loss", "ratio"}, set(out)
    assert abs(out["loss"].item() - EXAMPLE["loss"]) < 1e-12, out["loss"].item()
    assert torch.allclose(out["ratio"], torch.tensor(EXAMPLE["ratio"], dtype=torch.float64), atol=1e-12)
    assert out["loss"].dim() == 0 and out["loss"].dtype == torch.float64
    assert out["ratio"].dtype == torch.float64 and out["ratio"].shape == (2,)


check_example.example = True


def check_oracle(solve):
    torch.manual_seed(67)
    for _ in range(5):
        n, t = 4, 5
        lp = torch.randn(n, t, dtype=torch.float64) * 0.5
        old = torch.randn(n, t, dtype=torch.float64) * 0.5
        m = torch.rand(n, t) > 0.3
        m[:, 0] = True
        adv = torch.randn(n, dtype=torch.float64)
        out = solve(lp, old, m, adv, 0.2)
        ratios, loss = _oracle(lp.tolist(), old.tolist(), m.tolist(), adv.tolist(), 0.2)
        assert torch.allclose(out["ratio"], torch.tensor(ratios, dtype=torch.float64), atol=1e-12)
        assert abs(out["loss"].item() - loss) < 1e-12, (out["loss"].item(), loss)


def check_padding_invariance(solve):
    # mask=False 位置的 logp 无论多离谱都不进入序列比
    lp = torch.tensor([[0.3, 0.0, 0.0], [0.3, -50.0, 99.0]], dtype=torch.float64)
    old = torch.zeros_like(lp)
    m = torch.tensor([[True, True, True], [True, False, False]], dtype=torch.bool)
    adv = torch.tensor([0.5, 0.5], dtype=torch.float64)
    out = solve(lp, old, m, adv, 0.01)
    assert torch.allclose(out["ratio"], torch.tensor([math.exp(0.1), math.exp(0.3)], dtype=torch.float64), atol=1e-12)


def check_length_normalization(solve):
    # 同样的 delta 总和 0.3：摊到 3 个 token 得 s=e^0.1，摊到 1 个 token 得 s=e^0.3
    lp = torch.tensor([[0.1, 0.1, 0.1], [0.3, 0.0, 0.0]], dtype=torch.float64)
    m = torch.tensor([[True, True, True], [True, False, False]], dtype=torch.bool)
    out = solve(lp, torch.zeros_like(lp), m, torch.tensor([-1.0, -1.0], dtype=torch.float64), 3e-4)
    assert abs(out["ratio"][0].item() - math.exp(0.1)) < 1e-12
    assert abs(out["ratio"][1].item() - math.exp(0.3)) < 1e-12
    assert abs(out["loss"].item() - (math.exp(0.1) + math.exp(0.3)) / 2.0) < 1e-12


def check_sequence_level_clip(solve):
    # eps=0.01：正优势、s 高于上界 -> 整条序列一起裁到 1.01；负优势侧 min 选 s·A，保持不裁
    lp = torch.tensor([[0.3, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=torch.float64)
    m = torch.tensor([[True, True, True], [True, False, False]], dtype=torch.bool)
    out = solve(lp, torch.zeros_like(lp), m, torch.tensor([1.0, -1.0], dtype=torch.float64), 0.01)
    s3, s1 = math.exp(0.1), math.exp(0.3)
    assert abs(out["loss"].item() - (-(1.01 * 1.0 - s1) / 2.0)) < 1e-12, out["loss"].item()
    assert 1.01 < s3 < s1


def check_mean_over_answers_not_tokens(solve):
    # 回答长度 3 与 1、逐条 s 不同；按回答平均，而不是把所有 token 混在一起平均
    lp = torch.tensor([[0.3, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=torch.float64)
    m = torch.tensor([[True, True, True], [True, False, False]], dtype=torch.bool)
    out = solve(lp, torch.zeros_like(lp), m, torch.tensor([-1.0, -1.0], dtype=torch.float64), 3e-4)
    s3, s1 = math.exp(0.1), math.exp(0.3)
    per_answer = (s3 + s1) / 2.0
    per_token = (3.0 * s3 + s1) / 4.0
    assert abs(out["loss"].item() - per_answer) < 1e-12, out["loss"].item()
    assert abs(per_answer - per_token) > 0.05


def check_float32_input(solve):
    out = solve(LOGPS.to(torch.float32), torch.zeros_like(LOGPS, dtype=torch.float32),
                MASK, ADV.to(torch.float32), 3e-4)
    assert out["loss"].dtype == torch.float64 and out["ratio"].dtype == torch.float64
    assert abs(out["loss"].item() - EXAMPLE["loss"]) < 1e-6, out["loss"].item()


def check_errors(solve):
    lp = torch.zeros(2, 3, dtype=torch.float64)
    m = torch.ones(2, 3, dtype=torch.bool)
    adv = torch.zeros(2, dtype=torch.float64)

    def bad(fn, desc):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"应抛 ValueError: {desc}")

    bad(lambda: solve([[0.0, 0.0]], lp, m, adv), "logp 非张量")
    bad(lambda: solve(torch.zeros(2, 3, 1, dtype=torch.float64), lp, m, adv), "logp 非二维")
    bad(lambda: solve(lp, lp[:, :2], m, adv), "logp_old 形状不符")
    bad(lambda: solve(lp, lp, m.to(torch.int64), adv), "mask 非 bool")
    bad(lambda: solve(lp, lp, m[:, :2], adv), "mask 形状不符")
    bad(lambda: solve(lp, lp, torch.zeros(2, 3, dtype=torch.bool), adv), "有空回答")
    bad(lambda: solve(lp, lp, m, torch.zeros(3, dtype=torch.float64)), "adv 形状不符")
    bad(lambda: solve(lp, lp, m, adv.to(torch.float32)), "adv dtype 不符")
    bad(lambda: solve(lp, lp, m, adv, 0.0), "eps=0")
    bad(lambda: solve(lp, lp, m, adv, 1.0), "eps=1")
    bad(lambda: solve(torch.full((2, 3), float("inf"), dtype=torch.float64), lp, m, adv), "含 Inf")
