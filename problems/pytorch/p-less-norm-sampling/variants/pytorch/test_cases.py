import torch


def _oracle(logits):
    """独立重算：逐行 python 计算归一阈值并筛，与 solution 的张量路径不同。"""
    probs = torch.softmax(logits, dim=-1)
    v = probs.shape[-1]
    flat = probs.reshape(-1, v)
    out = torch.zeros_like(flat)
    for r in range(flat.shape[0]):
        p = [float(x) for x in flat[r]]
        s2 = sum(x * x for x in p)
        thr = (v * s2 - 1.0) / (v - 1.0)
        row = [x if x >= thr else 0.0 for x in p]
        tot = sum(row)
        out[r] = torch.tensor([x / tot for x in row])
    return out.reshape(probs.shape)


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    logits = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2])
    filtered, chosen = solve(logits, torch.tensor(0.5))
    exp = torch.softmax(logits, dim=-1)  # 全部保留 → filtered == probs
    assert torch.allclose(filtered, exp, atol=1e-5), filtered.tolist()
    assert chosen.item() == 2, chosen.item()


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
    # 均匀分布：thr = (v*(1/v)-1)/(v-1) = 0 → 全保留
    logits = torch.zeros(6)
    f, _ = solve(logits, torch.tensor(0.3))
    assert torch.allclose(f, torch.full((6,), 1 / 6), atol=1e-6), f.tolist()


def check_more_permissive_than_p_less(solve):
    # 同一分布，norm 保留数 >= 原版 p-less 保留数（thr_norm <= thr_pless）
    logits = torch.tensor([1.5, 1.2, 0.9, 0.3, 0.1, -0.5, -1.0, -1.4])
    probs = torch.softmax(logits, dim=-1)
    v = probs.shape[-1]
    s2 = float(probs.pow(2).sum())
    thr_pless = s2
    thr_norm = (v * s2 - 1.0) / (v - 1.0)
    n_pless = int((probs >= thr_pless).sum())
    n_norm = int((solve(logits, torch.tensor(0.3))[0] > 0).sum())
    assert n_norm >= n_pless, (n_norm, n_pless)


check_example.example = True
