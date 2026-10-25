"""参考答案：SigLIP 逐对 sigmoid 对比损失（分母为 batch 大小 B）。"""

import torch
import torch.nn.functional as F


def solve(sim, logit_scale, logit_bias):
    b = sim.shape[0]
    z = logit_scale * sim + logit_bias
    y = 2.0 * torch.eye(b, dtype=sim.dtype) - 1.0
    return -(F.logsigmoid(y * z).sum()) / b
