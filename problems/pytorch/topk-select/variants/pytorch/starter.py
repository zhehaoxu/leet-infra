import torch


def solve(x, k):
    # TODO: 沿最后一维取 top-k（降序），返回 (values, indices)。
    # 重复 k 次：argmax 取最大位置 → 记录值/索引 → 把该位置置 -inf 移出候选。
    raise NotImplementedError
