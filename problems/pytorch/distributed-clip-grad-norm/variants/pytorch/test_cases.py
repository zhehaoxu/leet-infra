import torch


def _ref(shards, max_norm, norm_type=2.0):
    """独立重算：把所有分片拼成一个大向量再算范数（与逐分片累加是不同路径）。"""
    allg = torch.cat(shards, dim=0).double()
    p = float(norm_type)
    total_norm = float((allg.abs() ** p).sum() ** (1.0 / p))
    scale = 1.0 if (total_norm == 0.0 or total_norm <= float(max_norm)) else float(max_norm) / total_norm
    return [g.double() * scale for g in shards], total_norm


def _tensors_close(a, b, rtol=1e-5):
    for x, y in zip(a, b):
        assert torch.allclose(x.double(), y.double(), rtol=rtol, atol=1e-8), (x.tolist(), y.tolist())


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    clipped, norm = solve([torch.tensor([3., 4.]), torch.tensor([0.])], 2.5)
    assert abs(norm - 5.0) < 1e-5, norm
    assert len(clipped) == 2
    assert torch.allclose(clipped[0], torch.tensor([1.5, 2.0])), clipped[0].tolist()
    assert torch.allclose(clipped[1], torch.tensor([0.0])), clipped[1].tolist()


def check_global_not_per_rank_independent(solve):
    # 全局范数=1.0（sqrt(0.36+0.64)），max_norm=0.5 -> scale=0.5
    # 若各 rank 独立裁（局部范数 0.6 / 0.8）会得到不同结果，这里必须是被同一个 0.5 缩放
    clipped, norm = solve([torch.tensor([0.6]), torch.tensor([0.8])], 0.5)
    assert abs(norm - 1.0) < 1e-5, norm
    assert torch.allclose(clipped[0], torch.tensor([0.3]), atol=1e-6), clipped[0].tolist()
    assert torch.allclose(clipped[1], torch.tensor([0.4]), atol=1e-6), clipped[1].tolist()


def check_no_clip_when_below(solve):
    clipped, norm = solve([torch.tensor([1., 1.])], 10.0)      # norm≈1.414 < 10
    assert abs(norm - (2 ** 0.5)) < 1e-5, norm
    assert torch.allclose(clipped[0], torch.tensor([1., 1.])), clipped[0].tolist()


def check_l1_norm(solve):
    clipped, norm = solve([torch.tensor([1., 2.]), torch.tensor([3.])], 3.0, norm_type=1.0)
    assert abs(norm - 6.0) < 1e-5, norm                        # |1|+|2|+|3|
    assert torch.allclose(clipped[0], torch.tensor([0.5, 1.0]), atol=1e-6)
    assert torch.allclose(clipped[1], torch.tensor([1.5]), atol=1e-6)


def check_zero_grad(solve):
    clipped, norm = solve([torch.tensor([0., 0.]), torch.tensor([0.])], 1.0)
    assert norm == 0.0, norm
    assert torch.equal(clipped[0], torch.zeros(2)) and torch.equal(clipped[1], torch.zeros(1))
    assert len(clipped[0]) == 2 and len(clipped[1]) == 1, "应保持各分片原形状"


def check_uneven_lengths_matches_ref(solve):
    torch.manual_seed(42)
    shards = [torch.randn(5), torch.randn(2), torch.randn(7)]
    for max_norm in [0.1, 3.0, 100.0]:
        clipped, norm = solve(shards, max_norm)
        want, wnorm = _ref(shards, max_norm)
        assert abs(norm - wnorm) <= 1e-5 * max(1.0, abs(wnorm)), (max_norm, norm, wnorm)
        assert len(clipped) == 3
        _tensors_close(clipped, want)


def check_input_not_mutated(solve):
    shards = [torch.tensor([3., 4.]), torch.tensor([0., 1., 2.])]
    snapshot = [t.clone() for t in shards]
    solve(shards, 1.0)
    for a, b in zip(shards, snapshot):
        assert torch.equal(a, b), "不应修改输入分片"


check_example.example = True
