import torch


def solve(logits, temp, delta, exponent, u):
    # TODO: Dynatemp 动态温度（u 为显式输入，保证确定性）。
    # probs=softmax；delta<=0 → dyn=temp, filtered=softmax(logits/temp)。
    # 否则 max_entropy=ln n, entropy=Σ-p ln p, C=entropy/max_entropy,
    # T_min=max(0,temp-delta), T_max=temp+delta, dyn=T_min+(T_max-T_min)*C**exponent,
    # filtered=softmax(logits/dyn)。
    # u 在 filtered CDF 逆变换得 chosen。返回 (filtered, chosen, dyn)。
    raise NotImplementedError
