"""参考答案：GAE 广义优势估计（反向递推）。"""

import torch


def solve(rewards, values, gamma, lam):
    T = rewards.shape[0]
    adv = torch.zeros(T)
    last = 0.0
    for t in reversed(range(T)):
        nxt = values[t + 1].item() if t + 1 < T else 0.0
        delta = rewards[t].item() + gamma * nxt - values[t].item()
        last = delta + gamma * lam * last
        adv[t] = last
    returns = adv + values
    return adv, returns
