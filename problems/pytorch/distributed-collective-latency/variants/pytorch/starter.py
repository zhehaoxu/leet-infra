import math


def solve(W: int, S: float, alpha: float, beta: float) -> dict:
    """Ring vs Recursive-Doubling All-Reduce 的 α-β 延迟账。

    W     : rank 数（2 的幂）
    S     : 每 rank 消息字节数
    alpha : 每步启动/延迟开销（秒）
    beta  : 每字节传输时间（秒/字节）
    返回 dict:
      ring_time = 2*(W-1)*(alpha + (S/W)*beta)
      rd_time   = log2(W)*(alpha + S*beta)
      faster    = "ring" 若 ring_time<=rd_time 否则 "recursive_doubling"
      speedup   = max/min（>=1）
    """
    # TODO: 按题面公式计算
    raise NotImplementedError
