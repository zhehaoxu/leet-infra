import torch
import torch.nn.functional as F


def _oracle(x, Wq, w_scale):
    # 物化反量化权重后 F.linear（与解法「先矩阵乘再按列 rescale」不同路径）
    W = Wq.to(x.dtype) * w_scale.to(x.dtype)[:, None]
    return F.linear(x, W)


def _assert(solve, x, Wq, w_scale):
    out = solve(x, Wq, w_scale)
    exp = _oracle(x, Wq, w_scale)
    assert out.shape == (x.shape[0], Wq.shape[0]), out.shape
    assert out.dtype == x.dtype, out.dtype
    assert torch.allclose(out, exp, rtol=1e-4, atol=1e-4), f"\n{out}\n期望\n{exp}"


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    x = torch.tensor([[1.0, 2.0, 0.0], [0.5, -1.0, 3.0]])
    Wq = torch.tensor([[10, -100, 50], [127, 0, -64]], dtype=torch.int8)
    w_scale = torch.tensor([0.01, 0.02])
    out = solve(x, Wq, w_scale)
    expected = torch.tensor([[-1.9, 2.54], [2.55, -2.57]])
    assert torch.allclose(out, expected, atol=1e-4), out.tolist()


def check_matches_oracle_random(solve):
    torch.manual_seed(42)
    for _ in range(20):
        M, N, K = 3, 4, 5
        x = torch.randn(M, K)
        Wq = torch.randint(-128, 128, (N, K), dtype=torch.int8)
        w_scale = torch.rand(N) * 0.02 + 0.001
        _assert(solve, x, Wq, w_scale)


def check_scale_direction(solve):
    # 每个输出通道独立 scale：把某通道 scale 置 0，该列输出应全 0
    x = torch.ones(1, 3)
    Wq = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int8)
    w_scale = torch.tensor([0.0, 1.0])
    out = solve(x, Wq, w_scale)
    assert out[0, 0].item() == 0.0
    assert abs(out[0, 1].item() - (4 + 5 + 6)) < 1e-4, out.tolist()


def check_single_channel(solve):
    x = torch.tensor([[2.0, 3.0]])
    Wq = torch.tensor([[10, -20]], dtype=torch.int8)
    out = solve(x, Wq, torch.tensor([0.1]))
    assert abs(out.item() - (2 * 10 + 3 * -20) * 0.1) < 1e-5, out.item()


def check_float64(solve):
    torch.manual_seed(1)
    x = torch.randn(2, 4, dtype=torch.float64)
    Wq = torch.randint(-128, 128, (3, 4), dtype=torch.int8)
    w_scale = torch.rand(3, dtype=torch.float64)
    _assert(solve, x, Wq, w_scale)


def check_does_not_mutate_input(solve):
    x = torch.randn(2, 3)
    Wq = torch.randint(-128, 128, (4, 3), dtype=torch.int8)
    w_scale = torch.rand(4)
    xs, Ws, ss = x.clone(), Wq.clone(), w_scale.clone()
    solve(x, Wq, w_scale)
    assert torch.equal(x, xs) and torch.equal(Wq, Ws) and torch.equal(w_scale, ss)


check_example.example = True
