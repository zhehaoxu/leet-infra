"""参考答案：SGD(+momentum) 单步 —— 严格复刻 torch.optim.SGD 更新式。"""

import torch


def solve(params, grads, buffers, lr, momentum=0.0, weight_decay=0.0):
    p = params.clone()
    g = grads.clone()
    b = buffers.clone()
    # 1) 权重衰减先并进梯度：grad += wd * param
    if weight_decay != 0:
        g = g + weight_decay * p
    # 2) 动量缓冲：buf = momentum * buf + grad（dampening=0）；无动量则直接用 grad
    if momentum != 0:
        b = momentum * b + g
        step_dir = b
    else:
        step_dir = g
    # 3) 参数更新：param -= lr * step_dir（nesterov=False）
    p = p - lr * step_dir
    return p, b
