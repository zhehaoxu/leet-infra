import torch
import torch.nn as nn


def _cell_params(B=2, In=3, H=5, seed=0):
    torch.manual_seed(seed)
    cell = nn.LSTMCell(In, H)
    x = torch.randn(B, In)
    h0 = torch.randn(B, H)
    c0 = torch.randn(B, H)
    return cell, x, h0, c0


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_zero_weights(solve):
    In, H = 3, 4
    w_ih = torch.zeros(4 * H, In)
    w_hh = torch.zeros(4 * H, H)
    b_ih = torch.zeros(4 * H)
    b_hh = torch.zeros(4 * H)
    x = torch.randn(1, In)
    h_prev = torch.zeros(1, H)
    c_prev = torch.full((1, H), 2.0)
    h, c = solve(x, h_prev, c_prev, w_ih, w_hh, b_ih, b_hh)
    assert torch.allclose(c, torch.ones(1, H), atol=1e-6), c.tolist()   # 0.5*2
    exp_h = 0.5 * torch.tanh(torch.ones(1, H))
    assert torch.allclose(h, exp_h, atol=1e-6), h.tolist()


def check_matches_lstmcell(solve):
    cell, x, h0, c0 = _cell_params(seed=0)
    h1, c1 = cell(x, (h0, c0))
    hs, cs = solve(x, h0, c0, cell.weight_ih, cell.weight_hh, cell.bias_ih, cell.bias_hh)
    assert torch.allclose(h1, hs, atol=1e-6) and torch.allclose(c1, cs, atol=1e-6)


def check_matches_lstmcell_other_seed(solve):
    cell, x, h0, c0 = _cell_params(B=4, In=6, H=8, seed=3)
    h1, c1 = cell(x, (h0, c0))
    hs, cs = solve(x, h0, c0, cell.weight_ih, cell.weight_hh, cell.bias_ih, cell.bias_hh)
    assert torch.allclose(h1, hs, atol=1e-6) and torch.allclose(c1, cs, atol=1e-6)


def check_shapes(solve):
    cell, x, h0, c0 = _cell_params(B=3, In=2, H=7, seed=1)
    h, c = solve(x, h0, c0, cell.weight_ih, cell.weight_hh, cell.bias_ih, cell.bias_hh)
    assert h.shape == (3, 7) and c.shape == (3, 7)


check_zero_weights.example = True
