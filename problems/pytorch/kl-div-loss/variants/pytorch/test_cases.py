import torch
import torch.nn.functional as F

# 示例：log_pred = log_softmax([[2,0,-1],[0,1,3]])，target = [[0.5,0.3,0.2],[0.1,0.6,0.3]]
# （各 reduction 的返回值实跑参考答案回填）
EXAMPLE_LOGITS = [[2.0, 0.0, -1.0], [0.0, 1.0, 3.0]]
EXAMPLE_TARGETS = [[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]]
EXAMPLE_NONE = [[-0.26165056228637695, 0.2897619903087616, 0.31208163499832153],
                [0.0867260992527008, 0.9954123497009277, -0.3102380335330963]]
EXAMPLE_BATCHMEAN = 0.556046724319458
EXAMPLE_SUM = 1.112093448638916


def check_example(solve):
    lp = F.log_softmax(torch.tensor(EXAMPLE_LOGITS), dim=1)
    t = torch.tensor(EXAMPLE_TARGETS)
    assert torch.allclose(solve(lp, t, reduction="none"), torch.tensor(EXAMPLE_NONE),
                          atol=1e-6)
    assert abs(float(solve(lp, t)) - EXAMPLE_BATCHMEAN) < 1e-6
    assert abs(float(solve(lp, t, reduction="sum")) - EXAMPLE_SUM) < 1e-6
    assert abs(float(solve(lp, t, reduction="mean")) - EXAMPLE_SUM / 6.0) < 1e-6


check_example.example = True


def check_matches_torch_dense(solve):
    torch.manual_seed(601)
    lp = F.log_softmax(torch.randn(4, 6), dim=1)
    tp = F.softmax(torch.randn(4, 6), dim=1)
    for red in ["none", "mean", "sum", "batchmean"]:
        ref = F.kl_div(lp, tp, reduction=red)
        got = solve(lp, tp, reduction=red)
        assert got.shape == ref.shape, (red, got.shape)
        assert torch.allclose(got, ref, atol=1e-6, rtol=1e-5), red
        assert got.dtype == torch.float32
    # log_target=True：target 传 log p
    ref = F.kl_div(lp, tp.log(), reduction="batchmean", log_target=True)
    got = solve(lp, tp.log(), reduction="batchmean", log_target=True)
    assert torch.allclose(got, ref, atol=1e-6, rtol=1e-5)


def check_batchmean_versus_mean(solve):
    # batchmean 只除 batch，mean 除 numel：C 类数 = 6 时两者相差恰为 C 倍
    torch.manual_seed(602)
    lp = F.log_softmax(torch.randn(4, 6), dim=1)
    tp = F.softmax(torch.randn(4, 6), dim=1)
    bm = float(solve(lp, tp, reduction="batchmean"))
    mn = float(solve(lp, tp, reduction="mean"))
    assert abs(bm - mn * 6.0) < 1e-5, (bm, mn)
    # 蒸馏里 KL 恒非负（同一批分布），而逐元素项可正可负
    assert bm >= 0.0
    assert solve(lp, tp, reduction="none").shape == (4, 6)


def check_zero_and_infinite_support(solve):
    # 硬标签 one-hot：p=0 处贡献恰为 0（题面口径 0·log 0 := 0），p=1 处退化成 -log q
    t = torch.tensor([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    lp = torch.log(F.softmax(torch.tensor([[1.0, 2.0, 0.5, -1.0], [3.0, -1.0, 0.0, 1.0]]), dim=1))
    got = solve(lp, t, reduction="none")
    assert torch.equal(got[0, [0, 2, 3]], torch.zeros(3)), got.tolist()
    assert torch.equal(got[1, [1, 2, 3]], torch.zeros(3))
    nll = -lp[0, 1]
    assert abs(float(got[0, 1]) - float(nll)) < 1e-6
    assert abs(float(solve(lp, t, reduction="batchmean"))
               - float((nll - lp[1, 0]) / 2.0)) < 1e-6
    # p>0 而 q=0：KL 为 +inf
    inf_out = solve(torch.tensor([[-float("inf"), 0.0]]), torch.tensor([[0.5, 0.5]]),
                    reduction="none")
    assert torch.isposinf(inf_out[0, 0]), inf_out.tolist()
    assert torch.isfinite(inf_out[0, 1])
    # p=0 且 q=0：按题面约定取 0（不复制 F.kl_div 的 NaN 实现细节）
    both0 = solve(torch.tensor([[-float("inf"), -1.0]]), torch.tensor([[0.0, 0.5]]),
                  reduction="none")
    assert float(both0[0, 0]) == 0.0, both0.tolist()


def check_broadcast_target(solve):
    # target 可与 log_pred 广播：同一共享目标分布作用于 batch 内每一步
    torch.manual_seed(603)
    lp = F.log_softmax(torch.randn(5, 4), dim=1)
    tp = F.softmax(torch.randn(4), dim=0)
    ref = F.kl_div(lp, tp, reduction="none")
    got = solve(lp, tp, reduction="none")
    assert got.shape == (5, 4)
    assert torch.allclose(got, ref, atol=1e-6, rtol=1e-5)
    assert solve(lp, tp, reduction="sum").shape == ()
    assert abs(float(solve(lp, tp, reduction="batchmean"))
               - float(solve(lp, tp, reduction="sum")) / 5.0) < 1e-6


def check_invalid_reduction(solve):
    lp = F.log_softmax(torch.randn(2, 3), dim=1)
    tp = F.softmax(torch.randn(2, 3), dim=1)
    for red in ["batchsum", "mean ", None, "MAX"]:
        try:
            solve(lp, tp, reduction=red)
        except ValueError:
            continue
        raise AssertionError(f"reduction={red!r} 应抛 ValueError")
