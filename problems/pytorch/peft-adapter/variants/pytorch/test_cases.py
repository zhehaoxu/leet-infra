import torch

X = torch.tensor([[0.61, 1.62, -0.80],
                  [0.52, -0.21, -0.07]])                    # (2,3) d=3
W_DOWN = torch.tensor([[0.4, -0.2, 0.1],
                       [0.0, 0.3, -0.5]])                   # (2,3) r=2
W_UP = torch.tensor([[0.2, 0.5],
                     [-0.3, 0.1],
                     [0.4, -0.2]])                          # (3,2)


def _oracle(x, W_down, W_up):
    """独立重算：逐样本列向量 W_down@x_i -> ReLU -> W_up@(.) + x_i（结构异于参考解的批量转置）。"""
    xs = x.reshape(-1, x.shape[-1])
    outs = []
    for xi in xs:
        h = torch.relu(W_down @ xi)
        outs.append(W_up @ h + xi)
    return torch.stack(outs).reshape(x.shape)


def check_example(solve):
    y = solve(X, W_DOWN, W_UP)
    exp = torch.tensor([[1.0529999732971191, 1.7086000442504883, -0.9772000312805176],
                        [0.5685999989509583, -0.28289997577667236, 0.02719999849796295]])
    assert torch.allclose(y, exp, atol=1e-5, rtol=1e-5), f"got {y.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(8)
    x = torch.randn(5, 7)
    wd = torch.randn(3, 7)
    wu = torch.randn(7, 3)
    y = solve(x, wd, wu)
    assert torch.allclose(y, _oracle(x, wd, wu), atol=1e-5, rtol=1e-5)


def check_zero_up_is_residual(solve):
    y = solve(X, W_DOWN, torch.zeros(3, 2))
    assert torch.allclose(y, X, atol=1e-6, rtol=1e-6), "W_up=0 时旁路为 0、输出应等于残差 x"


def check_bottleneck_relu_nonneg(solve):
    # 构造使下投影出现负分量：升维前 ReLU 应把它们截成 0，从而不影响上投影
    x = torch.tensor([[-1.0, -1.0, -1.0]])
    wd = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])   # down = [-1,-1] -> relu = [0,0]
    wu = torch.tensor([[5.0, 7.0], [1.0, 2.0], [3.0, 4.0]])  # (d=3, r=2)；瓶颈已为 0，up 恒为 0
    y = solve(x, wd, wu)
    assert torch.allclose(y, x, atol=1e-6, rtol=1e-6), "瓶颈负激活应被 ReLU 置零"


def check_3d_batch(solve):
    x3 = torch.stack([X, X.flip(0)])          # (2,2,3)
    y = solve(x3, W_DOWN, W_UP)
    assert y.shape == (2, 2, 3), f"3D 批量形状错: {tuple(y.shape)}"
    assert torch.allclose(y, _oracle(x3, W_DOWN, W_UP), atol=1e-5, rtol=1e-5)
