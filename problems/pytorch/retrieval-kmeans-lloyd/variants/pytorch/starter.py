import torch


def solve(points, init_centroids, tol=1e-4, max_iters=100):
    """k-means Lloyd 迭代：固定初始质心、逐轮分配/取均值，并对空簇重播种。

    入参：
    - points: (N, D) 浮点张量，dtype 为 float32 或 float64；N >= 1、D >= 1。
    - init_centroids: (K, D) 张量，dtype 必须与 points 相同；K >= 1 且
      K <= N。它是第 1 轮分配所用的初始质心，不使用任何随机初始化。
    - tol: float，默认 1e-4，收敛阈值，必须 >= 0。停止条件是
      「本轮新质心相对上一轮质心的逐质心欧氏位移的最大值 disp < tol」
      （严格小于）。tol=0 时即使 disp 恰好为 0 也不会提前停止，会跑满
      max_iters。
    - max_iters: int，默认 100，必须 >= 1，是允许执行的最大轮数。

    每一轮依次执行：
    1) 分配：每个点分配给平方欧氏距离最近的质心；距离并列时取质心下
       标最小者。
    2) 取均值：每个非空簇的新质心 = 该簇所有点逐维的算术均值。
    3) 空簇重播种：按空簇下标升序逐个处理。候选点是所有尚未在本轮被
       选作种子的点；按「该点到本轮分配所用旧质心的最近距离」降序排
       序，并列时点下标小者优先。取排序后的第一个候选点作为该空簇的
       新质心（即把离旧质心集合最远的点搬过去）。同一个点在同一轮内
       最多充当一个空簇的种子；若本轮有多个空簇，则依次取排序中的下
       一个候选点。重播种只改质心，不回溯修改本轮已算好的分配标签与
       均值；种子点仍然计入它本轮所属簇的均值。
    4) 停止：令 disp = max_j ||new_c[j] - old_c[j]||_2，若 disp < tol
       则在本轮结束后停止；否则继续。执行满 max_iters 轮也停止。返回
       的迭代次数是实际执行的轮数（包含触发收敛的最后一轮）。

    返回 (labels, centroids, n_iters)：
    - labels: (N,) int64。它用**返回的最终质心**重新做一次最近分配得
      到（并列取小下标），因此可能不同于最后一轮分配时使用的标签。
    - centroids: (K, D)，dtype 与 points 相同，是停止时的质心（已包含
      空簇重播种的结果）。
    - n_iters: Python int，实际执行的迭代轮数，1 <= n_iters <= max_iters。

    精度约定：距离与均值在 float64 下累加，最终质心 cast 回 points 的
    dtype，最终标签基于 cast 后的质心计算。

    非法输入一律抛 ValueError：points 或 init_centroids 不是二维张量、
    dtype 不是 float32/float64、两者 dtype 不同、特征维 D 不一致、
    N=0、K=0、K>N、max_iters 不是正整数、tol<0。
    """
    raise NotImplementedError
