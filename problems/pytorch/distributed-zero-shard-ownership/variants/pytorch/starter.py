import torch


def solve(param_numels, world):
    """ZeRO 一维分片归属。把所有参数展平成 total=sum(numels) 的一维 buffer，切成 world 份尽量
    等长的连续分片，rank r 独占全局下标半开区间 [floor(total*r/world), floor(total*(r+1)/world))。
    返回 (per_rank, straddle)：
      per_rank  长度 world 的 int64 张量，每个 rank 拥有的元素数（和为 total）；
      straddle  升序 int64 张量，跨越分片边界（参数区间 (s,e) 内有严格内部边界 b，s<b<e）的参数下标。
    world==1 时无内部边界，straddle 为空。"""
    # TODO
    raise NotImplementedError
