import math

import torch

W1 = [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]]
W2 = [0.3, 0.6]
X = [0.5, -1.0, 2.0]
Y = 1.0


def _oracle():
    """独立重算：纯 Python 手推前向与反向。
    tanh hook 把输出乘 2（前向账、反向账都要带上这个 2），
    fc1 的 full_backward_hook 把 grad_input 改写为 2*g+1（只影响 x.grad）。"""
    a1 = [sum(W1[i][j] * X[j] for j in range(3)) for i in range(2)]
    t = [math.tanh(v) for v in a1]
    h = [2.0 * v for v in t]                     # hook 替换后的输出
    out = sum(W2[j] * h[j] for j in range(2))
    loss = (out - Y) ** 2
    d_out = 2.0 * (out - Y)
    d_h = [W2[j] * d_out for j in range(2)]
    d_t = [2.0 * v for v in d_h]                 # 反向穿过 *2 替换
    d_a1 = [(1.0 - t[i] ** 2) * d_t[i] for i in range(2)]
    d_x = [sum(W1[i][j] * d_a1[i] for i in range(2)) for j in range(3)]
    x_grad_hooked = [2.0 * g + 1.0 for g in d_x]  # full_backward_hook 改写
    w1_grad = [[d_a1[i] * X[j] for j in range(3)] for i in range(2)]
    w2_grad = [d_out * h[j] for j in range(2)]
    return {
        "loss": loss,
        "a1": a1,
        "t": t,
        "fc2_input": h,
        "fc2_out": [out],
        "x_grad": x_grad_hooked,
        "fc1_weight_grad": [v for row in w1_grad for v in row],
        "fc2_weight_grad": w2_grad,
        "fc1_bias_grad": d_a1,
    }


EXP = _oracle()


def _close(a, b, name):
    assert torch.allclose(torch.tensor(a), torch.tensor(b), atol=1e-5, rtol=1e-5), \
        f"{name} 不符: got {a}, want {b}"


def check_example(solve):
    out = solve()
    _close(out["loss"], EXP["loss"], "loss")
    _close(out["acts"]["fc1"], EXP["a1"], "fc1 激活")
    _close(out["acts"]["tanh"], EXP["t"], "tanh 激活（替换前）")
    _close(out["fc2_input"], EXP["fc2_input"], "fc2 看到的输入（替换后 2*tanh）")
    assert out["fired"] == {"fc1_fwd": 1, "tanh_fwd": 1, "fc2_fwd": 1,
                            "fc1_bwd": 1, "fc2_bwd": 1}, f"hook 触发账不符: {out['fired']}"


check_example.example = True


def check_matches_oracle(solve):
    out = solve()
    _close(out["x_grad"], EXP["x_grad"], "x.grad（经 2*g+1 改写）")
    _close(out["fc1_weight_grad"], EXP["fc1_weight_grad"], "fc1.weight.grad")
    _close(out["fc2_weight_grad"], EXP["fc2_weight_grad"], "fc2.weight.grad")
    _close(out["fc1_bias_grad"], EXP["fc1_bias_grad"], "fc1.bias.grad")
    _close(out["loss"], EXP["loss"], "loss")


def check_hook_replacement_effect(solve):
    # 替换生效探针：fc2 的输入 = 2*tanh(a1)，而非 tanh(a1) 本身
    out = solve()
    t = out["acts"]["tanh"]
    seen = out["fc2_input"]
    assert all(abs(s - 2.0 * v) < 1e-5 for s, v in zip(seen, t)), \
        "forward hook 返回非 None 必须替换模块输出"
    _close(out["acts"]["fc2"], EXP["fc2_out"], "fc2 输出")


def check_param_grads_not_hooked(solve):
    # full_backward_hook 只改发给输入的梯度，参数梯度走原路：
    # 若改写泄漏进 weight.grad，与解析值必然对不上
    out = solve()
    hooked_into_w = [2.0 * g + 1.0 for g in EXP["fc1_bias_grad"]]
    assert not torch.allclose(torch.tensor(out["fc1_bias_grad"]),
                              torch.tensor(hooked_into_w), atol=1e-3), \
        "fc1.bias.grad 不应被 2*g+1 改写污染"
    _close(out["fc1_bias_grad"], EXP["fc1_bias_grad"], "fc1.bias.grad 保持解析值")


def check_x_grad_not_raw(solve):
    # x.grad 必须等于改写值而非原始值（防「忘 return」或「return None」的解法）
    out = solve()
    _close(out["x_grad"], EXP["x_grad"], "x.grad 应等于改写值")
    raw_dx = [(EXP["x_grad"][j] - 1.0) / 2.0 for j in range(3)]
    assert not torch.allclose(torch.tensor(out["x_grad"]), torch.tensor(raw_dx), atol=1e-3), \
        "x.grad 应是 2*g+1 改写后的值，不是原始梯度"
