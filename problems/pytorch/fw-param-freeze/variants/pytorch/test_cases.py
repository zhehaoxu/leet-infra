import math

import torch

# 题面固定的初始权重生成规则：_explicit(shape, start) = (arange(n)+start)/100
FLAGS = [
    ["encoder.0.weight", False],
    ["encoder.0.bias", False],
    ["encoder.2.weight", False],
    ["encoder.2.bias", False],
    ["head.weight", True],
    ["head.bias", True],
]


def _w(shape, start):
    """(arange(n)+start)/100 按行主序 reshape —— 与题面初始权重规则一致。"""
    rows, cols = shape[0], shape[1] if len(shape) == 2 else 1
    out = [[(i * cols + j + start) / 100.0 for j in range(cols)] for i in range(rows)]
    return out if len(shape) == 2 else [r[0] for r in out]


def _oracle():
    """独立重算：纯 Python 双循环做前向（math.tanh）、MSE 梯度与 SGD(momentum) 更新，
    完全不碰 torch.autograd / nn。"""
    W1 = _w((6, 8), 0)
    W2 = _w((6, 6), 48)
    x = [[i / 8.0 for i in range(b * 8, (b + 1) * 8)] for b in range(2)]
    y = [[(r * 5 + c) / 10.0 for c in range(5)] for r in range(2)]
    Wh = _w((5, 6), 0)          # 新 head（换头后从 0 起）
    bh = [0.0] * 5

    def lin(inp, W):            # out[b,i] = sum_j inp[b,j]*W[i,j]
        return [[sum(inp[b][j] * W[i][j] for j in range(len(W[0])))
                 for i in range(len(W))] for b in range(len(inp))]

    losses = []
    bufW = bufb = None
    for _ in range(3):
        t = [[math.tanh(v) for v in row] for row in lin(x, W1)]
        h = lin(t, W2)
        pred = [[sum(h[b][j] * Wh[i][j] for j in range(6)) + bh[i]
                 for i in range(5)] for b in range(2)]
        err = [[pred[b][i] - y[b][i] for i in range(5)] for b in range(2)]
        loss = sum(e * e for row in err for e in row) / 10.0
        losses.append(loss)
        # 梯度：dL/dW[i,j] = (2/10) * Σ_b err[b,i]*h[b,j]
        gWh = [[(2.0 / 10.0) * sum(err[b][i] * h[b][j] for b in range(2))
               for j in range(6)] for i in range(5)]
        gbh = [(2.0 / 10.0) * sum(err[b][i] for b in range(2)) for i in range(5)]
        # SGD momentum=0.9, lr=0.1: buf = 0.9*buf + g; w -= 0.1*buf
        if bufW is None:
            bufW, bufb = [row[:] for row in gWh], list(gbh)
        else:
            bufW = [[0.9 * bufW[i][j] + gWh[i][j] for j in range(6)] for i in range(5)]
            bufb = [0.9 * bufb[i] + gbh[i] for i in range(5)]
        Wh = [[Wh[i][j] - 0.1 * bufW[i][j] for j in range(6)] for i in range(5)]
        bh = [bh[i] - 0.1 * bufb[i] for i in range(5)]
    return losses, [v for row in Wh for v in row], bh


_EXP_LOSSES, _EXP_WH, _EXP_BH = _oracle()


def check_example(solve):
    out = solve()
    assert out["total_before"] == 124 and out["total_after"] == 131
    assert out["trainable_before"] == 0 and out["trainable_after"] == 35
    assert out["flags"] == FLAGS, f"requires_grad 标志账不符: {out['flags']}"
    assert out["frozen_bitwise_identical"] is True
    assert out["frozen_grad_is_none"] is True
    assert out["n_opt_state_params"] == 2
    for i, (g, e) in enumerate(zip(out["losses"], _EXP_LOSSES)):
        assert abs(g - e) <= 1e-5 + 1e-5 * abs(e), f"loss[{i}] got {g}, want {e}"


check_example.example = True


def check_matches_oracle(solve):
    out = solve()
    got_w = out["final_head_weight_flat"]
    got_b = out["final_head_bias_flat"]
    assert torch.allclose(torch.tensor(got_w), torch.tensor(_EXP_WH), atol=1e-5, rtol=1e-5), \
        "head.weight 终值与纯 Python 重算不符"
    assert torch.allclose(torch.tensor(got_b), torch.tensor(_EXP_BH), atol=1e-5, rtol=1e-5), \
        "head.bias 终值与纯 Python 重算不符"
    assert all(abs(g - e) <= 1e-5 + 1e-5 * abs(e) for g, e in zip(out["losses"], _EXP_LOSSES))


def check_frozen_params_untouched(solve):
    # 冻结的 encoder 参数逐位不变；可训 head 必须真的动了
    out = solve()
    init_w = [v / 100.0 for v in range(30)]
    assert any(a != b for a, b in zip(out["final_head_weight_flat"], init_w)), "head 应被更新"
    assert out["frozen_bitwise_identical"] is True


def check_trainable_only_in_optimizer(solve):
    # optimizer.state 只为传入的 2 个可训参数建缓冲区
    out = solve()
    assert out["n_opt_state_params"] == 2
    assert out["trainable_after"] == 35  # 5*6 + 5
    assert out["total_after"] - out["total_before"] == 7  # (6*5+5) - (6*4+4) = 35 - 28


def check_zero_grad_effect(solve):
    # 三步 loss 轨迹确定：首步是初始 head 下的 MSE，且非单调（动量过冲导致第 2 步回升）
    out = solve()
    assert len(out["losses"]) == 3
    assert out["losses"][1] > out["losses"][0], "动量过冲应使第 1 步 loss 回升"
