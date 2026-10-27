import torch


def solve(cache, queries, queue, alpha, beta):
    """把每个新请求派给一个副本：既要蹭上前缀缓存，又不能把某个副本堆爆。

    cache:(R, C) int64，第 r 行是副本 r 已缓存的前缀块哈希（-1 只是补位，不是哈希）;
    queries:(M, L) int64，第 j 行是请求 j 的前缀块哈希序列（遇到 -1 视为序列结束）;
    queue:(R,) int64，各副本当前的排队深度;alpha、beta 是两个代价权重（>= 0）。

    请求按行顺序逐个处理。对请求 j、副本 r：
      matched[j][r] = 请求前缀里、从头开始连续命中副本 r 缓存的块数（一旦断掉就停）;
      cost[r]       = alpha * (len_j - matched[j][r]) + beta * queue[r]。
    选 cost 最小的副本，并列取副本号小的；随后 queue[chosen] += 1，
    并把该请求的全部前缀块加入该副本的缓存（下次别人就能命中）。

    返回 (chosen:(M,)、hits:(M,)（各自命中的前缀块数）、
          final_queue:(R,)、total_cost:float32 标量（被选中副本的代价之和）)。
    """
    # TODO
    raise NotImplementedError
