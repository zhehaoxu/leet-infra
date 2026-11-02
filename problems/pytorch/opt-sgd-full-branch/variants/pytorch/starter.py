import torch


def solve(params, grads, lr, momentum=0.0, dampening=0.0, weight_decay=0.0,
          nesterov=False, maximize=False, buffers=None):
    # TODO: 复刻 torch.optim.SGD 的完整分支（多步）。
    #   - grads 里 None 表示该步无梯度：整步跳过（不建缓冲、不动参数）。
    #   - maximize：先把梯度取负，再并 weight_decay（顺序不能反）。
    #   - 首次建缓冲（buffers 为 None 且此前没建过）时 buf = g，忽略 dampening；
    #     已有缓冲（含调用方显式传入的全零缓冲）才走 mu*buf + (1-dampening)*g。
    #   - nesterov：step = g + momentum*buf（用刚更新的 buf）；否则 step = buf。
    #   - momentum == 0 时 step = g，缓冲全零，dampening 不参与。
    #   返回 {"params", "buffers", "path", "dirs"}；入参不得原地修改。
    raise NotImplementedError
