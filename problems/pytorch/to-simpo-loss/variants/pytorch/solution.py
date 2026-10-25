"""参考答案：SimPO 长度归一化偏好损失。"""

import torch
import torch.nn.functional as F


def solve(logp_w, logp_l, len_w, len_l, beta, gamma):
    r_w = logp_w / len_w
    r_l = logp_l / len_l
    margin = beta * (r_w - r_l) - gamma
    return -F.logsigmoid(margin).mean()
