import math

import torch

X = [-2.5, -0.75, 0.5, 1.25, 3.0]
C = 2.0
G = [1.0, 2.0, 3.0, 4.0, 5.0]


def _oracle():
    """独立重算：纯 Python math 层实现 y=c*x^2*tanh(x) 与解析梯度
    dy/dx = c*(2x*tanh x + x^2*(1-tanh^2 x))，再乘上游梯度 g。"""
    ys = [C * xv * xv * math.tanh(xv) for xv in X]
    gx = []
    for xv, gv in zip(X, G):
        t = math.tanh(xv)
        gx.append(gv * C * (2 * xv * t + xv * xv * (1 - t * t)))
    neg = sum(1 for xv in X if xv < 0)
    return ys, gx, neg


_EXP_Y, _EXP_GX, _EXP_NEG = _oracle()


def check_example(solve):
    out = solve()
    assert torch.allclose(torch.tensor(out["y_flat"]), torch.tensor(_EXP_Y),
                          atol=1e-5, rtol=1e-5), f"y 不符: {out['y_flat']}"
    assert out["neg_count"] == _EXP_NEG == 2
    assert out["neg_count_type"] == "int", "第二输出必须是纯 Python int 原样透出"
    assert out["y_has_grad_fn"] is True
    assert out["extra_grad_is_none"] is True, "int 输出不建图，backward 收到 None"
    assert out["needs_input_grad"] == [True, False], "c 不 requires_grad"
    assert out["grads_match_ref"] is True


check_example.example = True


def check_matches_oracle(solve):
    out = solve()
    assert torch.allclose(torch.tensor(out["grad_x_flat"]), torch.tensor(_EXP_GX),
                          atol=1e-5, rtol=1e-5), \
        f"手写 backward 与解析式不符: {out['grad_x_flat']}"
    assert torch.allclose(torch.tensor(out["grad_x_ref_flat"]), torch.tensor(_EXP_GX),
                          atol=1e-5, rtol=1e-5), "autograd 对照式梯度与解析式不符"
    assert torch.allclose(torch.tensor(out["y_flat"]), torch.tensor(_EXP_Y),
                          atol=1e-5, rtol=1e-5)


def check_call_counters(solve):
    # apply 两次（首次 + 复用检查），backward 入口被调用 3 次（第三次在取回已释放的
    # saved_tensors 时才抛 RuntimeError，入口计数已经发生）
    out = solve()
    assert out["fwd_calls"] == 2
    assert out["bwd_calls"] == 3


def check_second_backward_raises(solve):
    out = solve()
    assert out["second_backward_raises"] is True, \
        "save_for_backward 的张量在 backward 后被释放，重传必须报错"


def check_handwritten_equals_autograd(solve):
    # 「手写 backward vs autograd 对照」一致性用 torch.equal 级别的相对误差检查
    out = solve()
    a = torch.tensor(out["grad_x_flat"])
    b = torch.tensor(out["grad_x_ref_flat"])
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)
    # 且两者都不是平凡梯度 ones（grad_outputs 用了 arange(1,6)）
    assert not torch.allclose(a, torch.ones_like(a))
