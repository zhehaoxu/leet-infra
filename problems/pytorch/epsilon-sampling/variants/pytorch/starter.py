import torch


def solve(logits, eps, renorm, u):
    # TODO: Epsilon 采样（u 为显式输入，保证确定性）。
    # probs=softmax(logits) → thr = eps * probs.max(-1) → 保留 probs >= thr（等于阈值保留）→
    # renorm=True: 保留集合归一化；renorm=False: 保留项不缩放，把 1-sum(kept) 补给该行原始概率
    # 最大的 token（并列取索引最小者）。返回 (filtered, chosen)，chosen 用 u 在 filtered 原顺序
    # CDF 上逆变换采样。
    raise NotImplementedError
