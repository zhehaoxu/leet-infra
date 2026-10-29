import math

import torch

LOGPS = torch.tensor([
    [[0.20, -0.10, 0.05], [0.00, 0.10, -0.20]],
    [[0.05, 0.05, 0.05], [-0.05, 0.00, 0.10]],
], dtype=torch.float64)
LOGPS_OLD = torch.zeros_like(LOGPS)
MASK = torch.tensor([
    [[True, True, True], [True, False, True]],
    [[True, True, True], [True, True, True]],
], dtype=torch.bool)
REWARDS = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
EPS_LOW, EPS_HIGH = 0.2, 0.28
EXAMPLE = {
    "loss": -0.2717561038988343,
    "adv": [[1.0, -1.0], [0.0, 0.0]],
    "keep": [True, False],
    "num_tokens": 5,
}


def _oracle(logp, old, mask, rewards, eps_low, eps_high):
    b = len(logp)
    g = len(logp[0])
    t_len = len(logp[0][0])
    adv = []
    keep = []
    for i in range(b):
        rs = [rewards[i][j] for j in range(g)]
        mu = sum(rs) / g
        sd = math.sqrt(sum((v - mu) ** 2 for v in rs) / g)
        adv.append([0.0 if sd <= 1e-12 else (v - mu) / sd for v in rs])
        keep.append(0.0 < mu < 1.0)
    tot = 0.0
    n = 0
    for i in range(b):
        if not keep[i]:
            continue
        for j in range(g):
            for t in range(t_len):
                if not mask[i][j][t]:
                    continue
                rho = math.exp(logp[i][j][t] - old[i][j][t])
                c = min(max(rho, 1.0 - eps_low), 1.0 + eps_high)
                a = adv[i][j]
                tot += min(rho * a, c * a)
                n += 1
    return adv, keep, (0.0 if n == 0 else -tot / n), n


def check_example(solve):
    out = solve(LOGPS, LOGPS_OLD, MASK, REWARDS, EPS_LOW, EPS_HIGH)
    assert set(out) == {"loss", "adv", "keep", "num_tokens"}, set(out)
    assert abs(out["loss"].item() - EXAMPLE["loss"]) < 1e-12, out["loss"].item()
    assert out["adv"].tolist() == EXAMPLE["adv"], out["adv"].tolist()
    assert out["keep"].tolist() == EXAMPLE["keep"], out["keep"].tolist()
    assert out["num_tokens"].item() == EXAMPLE["num_tokens"]
    assert out["adv"].dtype == torch.float64
    assert out["keep"].dtype == torch.bool
    assert out["num_tokens"].dtype == torch.int64
    assert out["loss"].dim() == 0 and out["loss"].dtype == torch.float64


check_example.example = True


def check_oracle(solve):
    torch.manual_seed(53)
    for _ in range(5):
        b, g, t = 3, 4, 5
        lp = torch.randn(b, g, t, dtype=torch.float64) * 0.5
        old = torch.randn(b, g, t, dtype=torch.float64) * 0.5
        m = torch.rand(b, g, t) > 0.25
        m[:, :, 0] = True
        r = torch.randint(0, 2, (b, g)).to(torch.float64)
        out = solve(lp, old, m, r, 0.15, 0.3)
        adv, keep, loss, n = _oracle(lp.tolist(), old.tolist(), m.tolist(), r.tolist(), 0.15, 0.3)
        assert torch.allclose(out["adv"], torch.tensor(adv, dtype=torch.float64), atol=1e-12)
        assert out["keep"].tolist() == keep, (out["keep"].tolist(), keep)
        assert abs(out["loss"].item() - loss) < 1e-12, (out["loss"].item(), loss)
        assert out["num_tokens"].item() == n, (out["num_tokens"].item(), n)


def check_asymmetric_clip(solve):
    # 正优势、rho = exp(0.5) ≈ 1.6487：
    # 对称 eps=0.2 被裁到 1.2；非对称 (0.2, 0.8) 上界 1.8 不裁 -> 保留更大的策略改进
    lp = torch.full((1, 2, 1), 0.5, dtype=torch.float64)
    r = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    old = torch.zeros_like(lp)
    m = torch.ones(1, 2, 1, dtype=torch.bool)
    tight = solve(lp, old, m, r, 0.2, 0.2)
    wide = solve(lp, old, m, r, 0.2, 0.8)
    rho = math.exp(0.5)
    # 正优势序列贡献 1.2 vs 1.6487；负优势序列两边同为 min(rho*-1, clip*-1) = -rho
    assert abs(tight["loss"].item() - (-(1.2 - rho) / 2.0)) < 1e-12, tight["loss"].item()
    assert abs(wide["loss"].item() - (-(rho - rho) / 2.0)) < 1e-12, wide["loss"].item()
    assert wide["loss"].item() < tight["loss"].item(), "放宽上界应保留更大的 surrogate（损失更负）"


def check_lower_clip_still_active(solve):
    # rho = exp(-0.5) ≈ 0.6065 且 A = -1：正优势侧无关，负优势侧按 min 取更负的 clip·A
    lp = torch.full((1, 2, 1), -0.5, dtype=torch.float64)
    r = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    old = torch.zeros_like(lp)
    m = torch.ones(1, 2, 1, dtype=torch.bool)
    out = solve(lp, old, m, r, 0.2, 0.28)
    rho = math.exp(-0.5)
    # A=[-1, +1]：负优势项 = min(-rho, -clamp(rho,0.8,1.28)) = -0.8；正优势项 = rho
    assert abs(out["loss"].item() - (-(-0.8 + rho) / 2.0)) < 1e-12, out["loss"].item()
    assert rho < 0.8


def check_group_filter(solve):
    # 全对组 (acc=1) 与全错组 (acc=0) 都被丢弃：损失为 0，num_tokens 为 0，不产生 NaN
    lp = torch.full((2, 3, 2), 0.3, dtype=torch.float64)
    old = torch.zeros_like(lp)
    m = torch.ones(2, 3, 2, dtype=torch.bool)
    r = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    out = solve(lp, old, m, r, 0.2, 0.28)
    assert out["keep"].tolist() == [False, False]
    assert out["num_tokens"].item() == 0
    assert out["loss"].item() == 0.0
    assert torch.equal(out["adv"], torch.zeros(2, 3, dtype=torch.float64))


def check_filter_excludes_tokens(solve):
    # 保留组的 7 个有效 token 进分母，丢弃组的 9 个 token 不计入
    lp = torch.zeros(2, 3, 3, dtype=torch.float64)
    old = torch.zeros_like(lp)
    m = torch.ones(2, 3, 3, dtype=torch.bool)
    m[0, 2, 1] = False
    m[0, 2, 2] = False
    r = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    out = solve(lp, old, m, r, 0.2, 0.28)
    assert out["keep"].tolist() == [True, False]
    assert out["num_tokens"].item() == 7
    # rho 全为 1：保留组 adv = [1/√2, 1/√2, -√2]，7 个 token 的全局平均
    pos, neg = 1.0 / math.sqrt(2.0), -math.sqrt(2.0)
    expected = -(3.0 * pos + 3.0 * pos + neg) / 7.0
    assert abs(out["loss"].item() - expected) < 1e-12, (out["loss"].item(), expected)
    # 逐回答平均会得到 0，与 token 级聚合明显不同
    per_seq = -(pos + pos + neg) / 3.0
    assert abs(out["loss"].item() - per_seq) > 0.3


def check_global_token_mean(solve):
    # 全局 token 平均 != 逐回答平均：3 个有效 token 的正优势回答 + 1 个 token 的负优势回答
    lp = torch.tensor([[[0.0, 0.0, 0.0], [math.log(0.5), 0.0, 0.0]]], dtype=torch.float64)
    old = torch.zeros_like(lp)
    m = torch.tensor([[[True, True, True], [True, False, False]]], dtype=torch.bool)
    r = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    out = solve(lp, old, m, r, 0.2, 0.28)
    # 正优势：3 个 rho=1 的 token 各 +1；负优势：rho=0.5 被下界裁到 0.8，A=-1 -> -0.8
    assert abs(out["loss"].item() - (-(3.0 - 0.8) / 4.0)) < 1e-12, out["loss"].item()
    per_seq = -((1.0) + (-0.8)) / 2.0
    assert abs(out["loss"].item() - per_seq) > 0.3


def check_zero_std_group_adv(solve):
    # 混合组内 G=4、r=[1,1,0,0]：std=0.5，adv=±1；全 1 的组 adv 为 0
    lp = torch.zeros(2, 4, 1, dtype=torch.float64)
    old = torch.zeros_like(lp)
    m = torch.ones(2, 4, 1, dtype=torch.bool)
    r = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=torch.float64)
    out = solve(lp, old, m, r, 0.2, 0.28)
    assert torch.allclose(out["adv"][0], torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=torch.float64), atol=1e-12)
    assert torch.equal(out["adv"][1], torch.zeros(4, dtype=torch.float64))
    assert out["keep"].tolist() == [True, False]


def check_float32_input(solve):
    out = solve(LOGPS.to(torch.float32), LOGPS_OLD.to(torch.float32),
                MASK, REWARDS.to(torch.float32), EPS_LOW, EPS_HIGH)
    assert out["loss"].dtype == torch.float64 and out["adv"].dtype == torch.float64
    assert abs(out["loss"].item() - EXAMPLE["loss"]) < 1e-6, out["loss"].item()


def check_errors(solve):
    lp = torch.zeros(1, 2, 2, dtype=torch.float64)
    m = torch.ones(1, 2, 2, dtype=torch.bool)
    r = torch.tensor([[1.0, 0.0]], dtype=torch.float64)

    def bad(fn, desc):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"应抛 ValueError: {desc}")

    bad(lambda: solve([[0.0]], lp, m, r), "logp 非张量")
    bad(lambda: solve(lp, lp[0], m, r), "logp_old 非三维")
    bad(lambda: solve(lp, lp[:, :, :1], m, r), "logp_old 形状不符")
    bad(lambda: solve(lp, lp, m, torch.ones(1, 3, dtype=torch.float64)), "rewards 形状不符")
    bad(lambda: solve(lp, lp, m, torch.tensor([[1.0, 2.0]], dtype=torch.float64)), "rewards 非 0/1")
    bad(lambda: solve(lp, lp, m, torch.tensor([[1.0, 0.5]], dtype=torch.float64)), "rewards 非 0/1(0.5)")
    bad(lambda: solve(lp, lp, m.to(torch.int64), r), "mask 非 bool")
    bad(lambda: solve(lp, lp, torch.zeros(1, 2, 2, dtype=torch.bool), r), "有空回答")
    bad(lambda: solve(torch.zeros(1, 1, 2, dtype=torch.float64), torch.zeros(1, 1, 2, dtype=torch.float64),
                      torch.ones(1, 1, 2, dtype=torch.bool), torch.tensor([[1.0]], dtype=torch.float64)), "G=1")
    bad(lambda: solve(lp, lp, m, r, 1.0, 0.28), "eps_low=1")
    bad(lambda: solve(lp, lp, m, r, 0.2, -0.1), "eps_high<0")
    bad(lambda: solve(torch.full((1, 2, 2), float("nan"), dtype=torch.float64), lp, m, r), "含 NaN")
