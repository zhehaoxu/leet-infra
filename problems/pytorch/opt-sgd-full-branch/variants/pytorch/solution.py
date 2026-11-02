"""参考答案：torch.optim.SGD 全分支账（dampening / nesterov / maximize / 多步状态）。

逐行复刻 torch 2.14.1 `torch/optim/sgd.py`：`SGD.__init__` 的四道参数校验
（顺序敏感）+ `_single_tensor_sgd` 的单步路径。
"""

import torch


def _check(lr, momentum, weight_decay, dampening, nesterov):
    if lr < 0.0:
        raise ValueError(f"Invalid learning rate: {lr}")
    if momentum < 0.0:
        raise ValueError(f"Invalid momentum value: {momentum}")
    if weight_decay < 0.0:
        raise ValueError(f"Invalid weight_decay value: {weight_decay}")
    if nesterov and (momentum <= 0 or dampening != 0):
        raise ValueError("Nesterov momentum requires a momentum and zero dampening")


def solve(params, grads, lr, momentum=0.0, dampening=0.0, weight_decay=0.0,
          nesterov=False, maximize=False, buffers=None):
    _check(lr, momentum, weight_decay, dampening, nesterov)

    p = params.clone()
    buf = None if buffers is None else buffers.clone()
    path = [p.clone()]
    dirs = []

    for g_in in grads:
        if g_in is None:
            dirs.append(None)
            path.append(p.clone())
            continue

        g = g_in.clone()
        if maximize:
            g = -g
        if weight_decay != 0:
            g = g + weight_decay * p

        if momentum != 0:
            if buf is None:
                buf = g.clone()
            else:
                buf = momentum * buf + (1 - dampening) * g
            step = g + momentum * buf if nesterov else buf
        else:
            step = g

        p = p - lr * step
        dirs.append(step.clone())
        path.append(p.clone())

    if buf is None:
        buf = torch.zeros_like(params)

    return {"params": p, "buffers": buf, "path": path, "dirs": dirs}
