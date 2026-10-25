import torch


def solve(logits, mode, drip_amount, min_keep, u):
    # TODO: Dripping 采样（砍超高概率头部；u 为显式输入）。
    # probs=softmax → valid = probs > 1e-35 → mean、总体 std（分母 n）→
    # mode="mean": thr = drip_amount*mean；mode="std": thr = mean + drip_amount*std →
    # cut = probs > thr；若无 token 被砍 或 存活数 < min_keep → 整行不砍 →
    # 存活集合归一化得 filtered（原顺序）→ u 在原顺序 CDF 逆变换得 chosen。
    # 返回 (filtered, chosen, cut_count)，cut_count 为 int64 张量 (...)，未触发时为 0。
    raise NotImplementedError
