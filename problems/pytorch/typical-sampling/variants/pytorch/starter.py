import torch


def solve(logits, tau, u):
    # TODO: Typical / 局部典型采样（u 为显式输入，保证确定性）。
    # probs=softmax → H=Σ-p·ln p → d=|(-ln p)-H| → 按 d 升序(并列原索引)累积概率到首次>tau →
    # 保留该集合、原顺序归一化得 filtered → 用 u 在 filtered CDF 逆变换采样得 chosen。
    # 返回 (filtered, chosen)。注意：可能丢弃 argmax。
    raise NotImplementedError
