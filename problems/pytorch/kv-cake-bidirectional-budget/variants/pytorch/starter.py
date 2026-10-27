import torch


def solve(scores, budget, edges):
    """CAKE：按层偏好分配全局预算 + 沿依赖边级联传递重要性。
    scores:(Lc,L) float32 非负，逐层累计重要性；budget 为全模型总保留位；
    edges 为 [(src,dst)] 层依赖（按给定顺序处理：work[dst] += work[src]，可传递）。
    1) 预分配 n_l = floor(budget*mass_l/total)，再按小数部分降序（并列取小层号）
       逐个 +1 分完余量，单层不超过 L；总分为 0 时全部 0。
    2) 传递后每层取 work 的 top-n_l（并列取小下标）。
    返回 (alloc:(Lc,) int64, keep:(Lc,L) bool)。
    参数非法（budget<0、负分数、自环、层号越界）抛 ValueError。"""
    # TODO
    raise NotImplementedError
