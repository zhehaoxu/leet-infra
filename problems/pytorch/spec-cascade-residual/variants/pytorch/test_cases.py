import numpy as np
import torch


def _oracle(p, qs):
    """numpy 逐级补齐算 accepted_mass，再用 Σacc + |rem|·p 独立拼出 out。"""
    target = p.numpy().astype(np.float64)
    drafts = qs.numpy().astype(np.float64)
    rem = target.copy()
    hit = np.zeros_like(target)
    accepted = []
    for level in range(drafts.shape[0]):
        take = np.minimum(drafts[level], rem)
        hit = hit + take
        rem = rem - take
        accepted.append(float(take.sum()))
    fallback = float(rem.sum())
    return hit + fallback * target, np.array(accepted, dtype=np.float64), fallback


def check_example(solve):
    torch.manual_seed(0)
    p = torch.tensor([0.5, 0.3, 0.2])
    qs = torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    out, accepted, target_mass = solve(p, qs)
    assert torch.allclose(out, p, atol=1e-6), f"两级正好补满 → out 应等于 p，got {out.tolist()}"
    assert torch.allclose(accepted, torch.tensor([0.8, 0.2]), atol=1e-6), (
        f"got {accepted.tolist()}")
    assert abs(float(target_mass)) < 1e-6, f"target 应无残余，got {float(target_mass)}"


check_example.example = True


def check_exact_and_duplicate_levels(solve):
    """与 target 同分布的一级就吃满；同一分布重复两级时功劳归第一级。"""
    torch.manual_seed(1)
    p = torch.tensor([0.4, 0.35, 0.25])
    qs = torch.stack([p, torch.tensor([0.1, 0.45, 0.45]), torch.tensor([0.2, 0.3, 0.5])])
    out, accepted, target_mass = solve(p, qs)
    assert torch.allclose(out, p, atol=1e-6), f"got {out.tolist()}"
    assert torch.allclose(accepted, torch.tensor([1.0, 0.0, 0.0]), atol=1e-6), (
        f"got {accepted.tolist()}")
    assert abs(float(target_mass)) < 1e-6

    torch.manual_seed(2)
    p = torch.tensor([0.5, 0.3, 0.2])
    q = torch.tensor([0.4, 0.4, 0.2])
    out, accepted, target_mass = solve(p, torch.stack([q, q]))
    assert torch.allclose(out, p, atol=1e-6), f"got {out.tolist()}"
    assert torch.allclose(accepted, torch.tensor([0.9, 0.1]), atol=1e-6), (
        f"第一级应贡献 0.9、第二级只补剩下的 0.1，got {accepted.tolist()}")
    assert abs(float(target_mass)) < 1e-6


def check_insufficient_coverage_falls_back(solve):
    """所有级加起来不足以覆盖 target → 命中只到 Σq，残余质量按 p 重新分配给 target。"""
    torch.manual_seed(3)
    p = torch.tensor([0.5, 0.3, 0.2])
    qs = torch.tensor([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]])
    out, accepted, target_mass = solve(p, qs)
    assert torch.allclose(accepted, torch.tensor([0.6, 0.3]), atol=1e-6), (
        f"got {accepted.tolist()}")
    assert abs(float(target_mass) - 0.1) < 1e-6, f"got {float(target_mass)}"
    want = torch.tensor([0.5, 0.3, 0.1]) + 0.1 * p
    assert torch.allclose(out, want, atol=1e-6), f"got {out.tolist()}"
    assert abs(float(out.sum()) - 1.0) < 1e-6, "out 必须仍然是分布"
    assert not torch.allclose(out, p, atol=1e-4), "有回退时 out 不应等于 p（这会引入偏差）"


def check_order_changes_credit_not_output(solve):
    """换序不改变 out（只取决于 Σq 与 p），但改变每一级拿到的功劳。"""
    torch.manual_seed(4)
    p = torch.tensor([0.5, 0.3, 0.2])
    q1 = torch.tensor([0.9, 0.05, 0.05])
    q2 = torch.tensor([0.05, 0.9, 0.05])
    out_a, acc_a, tm_a = solve(p, torch.stack([q1, q2]))
    out_b, acc_b, tm_b = solve(p, torch.stack([q2, q1]))
    assert torch.allclose(out_a, out_b, atol=1e-6), "out 应与级联顺序无关"
    assert abs(float(tm_a) - float(tm_b)) < 1e-6, "残余质量应与级联顺序无关"
    assert not torch.allclose(acc_a, acc_b, atol=1e-4), "每级贡献必须与顺序相关"
    assert torch.allclose(acc_a, torch.tensor([0.6, 0.3]), atol=1e-6), f"got {acc_a.tolist()}"
    assert torch.allclose(acc_b, torch.tensor([0.4, 0.5]), atol=1e-6), f"got {acc_b.tolist()}"


def check_matches_oracle_and_errors(solve):
    """随机分布对拍 + 质量守恒 + 非法输入。"""
    torch.manual_seed(21)
    for vocab, levels in ((3, 2), (8, 4), (1, 3)):
        p = torch.rand(vocab)
        p = p / p.sum()
        qs = torch.rand(levels, vocab)
        qs = qs / qs.sum(dim=1, keepdim=True)
        out, accepted, target_mass = solve(p, qs)
        exp_out, exp_acc, exp_tm = _oracle(p, qs)
        assert torch.allclose(out, torch.tensor(exp_out, dtype=torch.float32), atol=1e-6), (
            f"{(vocab, levels)} out 不符")
        assert torch.allclose(accepted, torch.tensor(exp_acc, dtype=torch.float32), atol=1e-6), (
            f"{(vocab, levels)} accepted_mass 不符")
        assert abs(float(target_mass) - exp_tm) < 1e-6, f"{(vocab, levels)} target_mass 不符"
        assert abs(float(out.sum()) - 1.0) < 1e-5, "out 必须仍然是分布（质量守恒）"
        assert abs(float(accepted.sum()) + float(target_mass) - 1.0) < 1e-5, (
            "各级贡献 + target 残余 = 1")

    p = torch.tensor([0.5, 0.5])
    good = torch.tensor([[0.5, 0.5]])
    bad_calls = [
        (torch.tensor([0.6, 0.5]), good),                  # p 未归一化
        (p, torch.tensor([[0.6, 0.5]])),                   # q 未归一化
        (p, torch.tensor([[1.2, -0.2]])),                  # 负概率
        (p, torch.tensor([[0.5, 0.3, 0.2]])),              # 词表大小不一致
        (torch.tensor([[0.5, 0.5]]), good),                # p 非一维
        (p, torch.tensor([0.5, 0.5])),                     # qs 非二维
    ]
    for args in bad_calls:
        try:
            solve(*args)
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
