import torch


def _oracle(logits):
    """独立重算：逐行 python 计算 Σp² 阈值并筛，与 solution 的张量 keep/mask 路径不同。"""
    probs = torch.softmax(logits, dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    out = torch.zeros_like(flat)
    for r in range(flat.shape[0]):
        p = [float(x) for x in flat[r]]
        thr = sum(v * v for v in p)
        row = [v if v >= thr else 0.0 for v in p]
        s = sum(row)
        out[r] = torch.tensor([v / s for v in row])
    return out.reshape(probs.shape)


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    logits = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2])
    filtered, chosen = solve(logits, torch.tensor(0.7))
    exp = torch.tensor([0.549834, 0.450166, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert torch.allclose(filtered, exp, atol=1e-5), filtered.tolist()
    assert chosen.item() == 1, chosen.item()


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for _ in range(50):
        logits = torch.randn(torch.randint(4, 16, (1,)).item())
        f, _ = solve(logits, torch.tensor(0.3))
        assert torch.allclose(f, _oracle(logits), atol=1e-6)


def check_sum_and_in_set(solve):
    torch.manual_seed(1)
    logits = torch.randn(5, 12)
    f, c = solve(logits, torch.rand(5))
    assert torch.allclose(f.sum(-1), torch.ones(5), atol=1e-6)
    assert bool((f.gather(-1, c.view(-1, 1)) > 0).all())
    assert int((f[0] > 0).sum()) >= 1


def check_uniform_keeps_all(solve):
    # 全等概率分布：Σp² = 1/V，每个 p_i = 1/V ≥ thr → 全保留 → filtered == probs
    logits = torch.zeros(6)
    f, _ = solve(logits, torch.tensor(0.3))
    assert torch.allclose(f, torch.full((6,), 1 / 6), atol=1e-6), f.tolist()


def check_sharp_keeps_argmax(solve):
    # 很尖的分布：只留 argmax
    logits = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0])
    f, c = solve(logits, torch.tensor(0.3))
    assert torch.allclose(f, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]), atol=1e-6), f.tolist()
    assert c.item() == 0


check_example.example = True
