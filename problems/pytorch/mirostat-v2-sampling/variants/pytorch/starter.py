import torch


def solve(logits, tau, eta, mu, u):
    # TODO: Mirostat v2（mu 状态显式入/出；u 为显式输入，保证确定性）。
    # probs=softmax → 按 p 降序，截断保留前缀直到 -log2(p) > mu（size=max(k,1)）→
    # 归一化映射回原顺序得 filtered → u 在 filtered CDF 逆变换得 chosen →
    # obs = -log2(filtered[chosen])；mu_new = mu - eta*(obs - tau)。
    # 返回 (filtered, chosen, mu_new)。
    raise NotImplementedError
