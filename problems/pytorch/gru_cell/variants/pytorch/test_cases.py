import torch
import torch.nn as nn


def _cell_ref(x, h, w_ih, w_hh, b_ih, b_hh):
    H = h.shape[1]
    I = x.shape[1]
    cell = nn.GRUCell(I, H)
    with torch.no_grad():
        cell.weight_ih.copy_(w_ih)
        cell.weight_hh.copy_(w_hh)
        cell.bias_ih.copy_(b_ih)
        cell.bias_hh.copy_(b_hh)
    return cell(x, h)


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_matches_torch(solve):
    torch.manual_seed(0)
    for I, H, B in [(2, 3, 4), (5, 8, 2), (3, 3, 6)]:
        x = torch.randn(B, I)
        h = torch.randn(B, H)
        w_ih = torch.randn(3 * H, I)
        w_hh = torch.randn(3 * H, H)
        b_ih = torch.randn(3 * H)
        b_hh = torch.randn(3 * H)
        got = solve(x, h, w_ih, w_hh, b_ih, b_hh)
        assert torch.allclose(got, _cell_ref(x, h, w_ih, w_hh, b_ih, b_hh), atol=1e-6), f"I={I} H={H}"


def check_edge_example(solve):
    x = torch.tensor([[0.5, -0.5]])
    h = torch.tensor([[0.2, -0.3, 0.1]])
    w_ih = torch.arange(18, dtype=torch.float32).view(9, 2) / 20 - 0.4
    w_hh = torch.arange(27, dtype=torch.float32).view(9, 3) / 20 - 0.4
    z = torch.zeros(9)
    got = solve(x, h, w_ih, w_hh, z, z)
    ref = torch.tensor([[0.084566, -0.161684, 0.035316]])
    assert torch.allclose(got, ref, atol=1e-5), got.tolist()


def check_output_shape(solve):
    torch.manual_seed(1)
    I, H, B = 4, 7, 3
    x = torch.randn(B, I)
    h = torch.randn(B, H)
    out = solve(x, h, torch.randn(3 * H, I), torch.randn(3 * H, H), torch.randn(3 * H), torch.randn(3 * H))
    assert out.shape == (B, H)


def check_zero_update_gate_keeps_history(solve):
    # 手写极端权重使 z≈1（几乎全保留旧 h），h' 应接近 h
    I, H, B = 2, 3, 1
    x = torch.zeros(B, I)
    h = torch.tensor([[0.7, -0.2, 0.4]])
    w_ih = torch.zeros(3 * H, I)
    w_hh = torch.zeros(3 * H, H)
    b_ih = torch.zeros(3 * H)
    b_hh = torch.zeros(3 * H)
    # update 段 bias 设大 → z≈1；new 段 bias 设 0
    b_hh[H:2 * H] = 10.0
    got = solve(x, h, w_ih, w_hh, b_ih, b_hh)
    assert torch.allclose(got, h, atol=1e-2), got.tolist()


check_matches_torch.example = True
