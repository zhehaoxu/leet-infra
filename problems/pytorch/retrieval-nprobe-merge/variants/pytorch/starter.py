import torch


def solve(queries, centroids, invlist_vectors, invlist_ids,
          nprobe=1, k=10):
    """nprobe 多探测与跨 cell 归并（倒排表存完整向量，距离为平方欧氏）。

    入参：
    - queries: (Q, D) 浮点张量，float32 或 float64，Q 可以为 0。
    - centroids: (C, D) 张量，与 queries 同 dtype，C 个粗量化中心；C 可以为 0。
    - invlist_vectors: 长度 C 的 list，第 c 项是 (n_c, D) 张量，dtype 与
      queries 相同，存 cell c 内文档的完整向量；n_c 可以为 0。
    - invlist_ids: 长度 C 的 list，第 c 项是 (n_c,) int64 张量，与
      invlist_vectors[c] 逐行对应，是文档的全局 id；要求 id >= 0。同一个
      全局 id 允许在不同 cell（甚至同一 cell）出现多次，此时按“去重取最小
      距离”处理。
    - nprobe: int，默认 1；每个查询按“query 到粗中心的平方欧氏距离升序”
      选前 nprobe 个 cell，距离并列时取 cell 号小者。nprobe < 0 抛
      ValueError；nprobe > C 视为探测全部 C 个 cell；nprobe == 0 表示不探测
      任何 cell。
    - k: int，默认 10；每个查询返回的槽位数。k < 0 抛 ValueError；
      k == 0 时 ids/dists 的形状为 (Q, 0)。

    距离与归并口径：
    - 单条 posting 的距离 = || query - invlist_vectors[c][i] ||^2，内部用
      float64 累加（逐维平方差求和），输出时转回 queries.dtype。
    - 按选中的 cell 顺序（不是按 cell 号排序）依次探测，把各 cell 的
      posting 取并集；同一全局 id 出现多次时只保留一次，距离取这些出现中
      的最小值。
    - 去重后的候选按 (距离升序, 全局 id 升序) 排列，取前 k 个。

    返回 (ids, dists, cells, stats)：
    - ids: (Q, k) int64。真实候选不足 k 个时尾部填 -1。
    - dists: (Q, k)，dtype 与 queries 相同，与 ids 逐位对齐；填充位为 +inf。
    - cells: (Q, P) int64，P = min(nprobe, C)；每行是实际探测的 cell 号序列，
      按“query 到该 cell 中心的平方距离升序、并列取小 cell 号”排列。
      nprobe == 0 时形状为 (Q, 0)；nprobe > C 时 P == C。
    - stats: (Q, 4) int64，四列依次为
      [n_postings, n_unique, n_duplicates, n_returned]：
      * n_postings：被探测 cell 里的 posting 总数（含重复 id）；
      * n_unique：其中不同全局 id 的个数；
      * n_duplicates：n_postings - n_unique；
      * n_returned：真正写入 ids 的真实候选数，等于 min(k, n_unique)。
      没有探测到任何 posting 时四列全 0。

    错误约定（以下情况一律抛 ValueError，消息内容不限）：
    - queries/centroids 的维度不是 2；D <= 0；centroids 第二维不是 D。
    - queries 不是 float32/float64，或 centroids 与 queries dtype 不同。
    - invlist_vectors 或 invlist_ids 的长度不等于 C。
    - 某个 cell 的 vectors 不是 (n_c, D)、dtype 与 queries 不同；
      ids 不是 1-D int64；ids 与 vectors 行数不同。
    - 任一全局 id < 0。
    - nprobe < 0 或 k < 0。
    """
    raise NotImplementedError
