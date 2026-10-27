import torch


def solve(boundaries, x, right=False, weights=None):
    """searchsorted 分桶 + bincount 直方图：先逐元素查桶，再对桶下标做计数。

    入参：
    - boundaries: (M,) 一维张量，已按升序排列（可含重复值），M = 0 允许；
      dtype 与 x 一致（int64 或 float32/float64 均支持）。
    - x: 任意形状的张量，dtype 与 boundaries 一致。
    - right: bool，默认 False，逐元素 searchsorted 的口径（与
      torch.searchsorted 的 side 一致，经实测定稿）：
      * right=False：返回最小下标 i 使 boundaries[i] >= x，即严格小于 x 的
        boundary 个数；所有 boundary 都 < x 时返回 M。
      * right=True：返回最小下标 i 使 boundaries[i] > x，即 <= x 的 boundary
        个数；所有 boundary 都 <= x 时返回 M。
    - weights: 可选，与 x 同形状的浮点张量（float32 / float64）。给出时做
      加权 bincount：counts[i] = 落在桶 i 的元素 weight 之和，counts 为浮点
      （dtype 跟随 weights）；不给时 counts 为 int64 计数。

    返回 (idx, counts) 二元组：
    - idx: 与 x 同形状的 int64 张量，每个元素的桶下标，取值范围 [0, M]。
      桶语义：桶 i（1 <= i <= M-1）对应左闭右开段 [boundaries[i-1],
      boundaries[i])，桶 0 为 (-inf, boundaries[0])，桶 M 为
      [boundaries[M-1], +inf)，首尾为开区间段（含重复 boundary 时以 idx 的
      上述定义为准）。
    - counts: 长度恰为 M+1 的一维张量，对 idx 做 bincount（counts[i] =
      idx 等于 i 的元素个数，或对应 weight 之和）。x 为空张量时 idx 为空、
      counts 全 0。
    """
    # TODO
    raise NotImplementedError
