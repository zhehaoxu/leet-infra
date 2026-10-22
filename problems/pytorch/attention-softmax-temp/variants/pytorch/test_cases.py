import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.1]])
    y = solve(logits, temperature=1.0)
    exp = torch.softmax(logits, dim=-1)
    assert torch.allclose(y, exp, atol=1e-6), f"got {y.tolist()}, expected {exp.tolist()}"


def check_temperature_effect(solve):
    torch.manual_seed(42)
    logits = torch.randn(4, 10)
    cold = solve(logits, temperature=0.01)
    hot = solve(logits, temperature=50.0)
    # T→0 趋近 one-hot(argmax)；T→∞ 趋近均匀分布
    argmax_onehot = torch.zeros_like(cold)
    argmax_onehot.scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
    assert torch.allclose(cold, argmax_onehot, atol=1e-4), "低温应趋近 argmax one-hot"
    assert torch.allclose(hot, torch.full_like(hot, 0.1), atol=1e-2), "高温应趋近均匀"


def check_temperature_order(solve):
    # 温度作用于 softmax 之前：T<1 放大差距（最大概率 > T=1 时）
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    p1 = solve(logits, 1.0)[0, 2].item()
    p_low = solve(logits, 0.5)[0, 2].item()
    p_high = solve(logits, 2.0)[0, 2].item()
    assert p_low > p1 > p_high, f"概率集中顺序错误: {p_low}, {p1}, {p_high}"


def check_top_p_nucleus(solve):
    # 头部集中分布（前两项概率和≈1）：top_p=0.9 保留前两项，top_p=0.5 只保留第一项
    logits = torch.tensor([[10.0, 9.0, 0.0, -5.0]])
    full = solve(logits, 1.0)
    assert float(full[0, 0]) + float(full[0, 1]) > 0.99, "softmax 计算错误"
    y = solve(logits, 1.0, top_p=0.9)
    assert y[0, 2] == 0.0 and y[0, 3] == 0.0, "尾部应被截断"
    assert y[0, 0] > 0 and y[0, 1] > 0, "nucleus 项应保留"
    assert torch.allclose(y.sum(-1), torch.ones(1), atol=1e-6), "nucleus 后应重归一化"
    assert torch.allclose(y[0, :2], full[0, :2] / full[0, :2].sum(), atol=1e-6), "保留项相对比例应保持"
    y2 = solve(logits, 1.0, top_p=0.5)
    assert y2[0, 0] == 1.0 and y2[0, 1:].sum() == 0.0, "单项概率 0.73 已超 top_p=0.5，应只留一项"


def check_top_p_at_least_one(solve):
    # top_p 小于最大单项概率时也必须保留至少 1 项
    probs = solve(torch.tensor([[100.0, 0.0, 0.0]]), 1.0, top_p=0.01)
    assert torch.isfinite(probs).all() and probs.sum(-1).allclose(torch.ones(1))
    assert float(probs[0, 0]) == 1.0, "应只剩 argmax 项"


def check_batched_rows(solve):
    torch.manual_seed(1)
    logits = torch.randn(3, 5, 8)
    y = solve(logits, 2.0, 0.8)
    assert y.shape == logits.shape
    assert torch.allclose(y.sum(-1), torch.ones(3, 5), atol=1e-6), "每行概率和为 1"
    assert (y >= 0).all(), "概率非负"
    # 每行被置零的都是该行最小概率项
    for b in range(3):
        for r in range(5):
            kept_mask = y[b, r] > 0
            dropped = y[b, r][~kept_mask]
            if dropped.numel():
                full_row = torch.softmax(logits[b, r] / 2.0, -1)
                assert full_row[~kept_mask].max() <= full_row[kept_mask].min() + 1e-12, "应截掉最小的项"


def check_stable_large_logits(solve):
    y = solve(torch.tensor([[1000.0, 999.0, 998.0]]))
    assert torch.isfinite(y).all() and y.sum(-1).allclose(torch.ones(1)), "大 logits 需减 max"


check_basic.example = True
