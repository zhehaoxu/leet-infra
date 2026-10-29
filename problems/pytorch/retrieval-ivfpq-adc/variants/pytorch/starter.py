import torch


def solve(queries, centroids, codebooks, invlist_ids, invlist_codes,
          nprobe=1, k=10):
    """IVF-PQ 非对称 ADC 打分：查询不量化，候选只存 PQ 码字。

    入参：
    - queries: (Q, D) 浮点张量，float32 或 float64，Q 可以为 0。查询向量
      保持原精度，不做 PQ 编码（这就是 asymmetric 的含义）。
    - centroids: (C, D) 张量，与 queries 同 dtype，C 个粗量化中心；C 可以为 0。
    - codebooks: (m, K, d) 张量，与 queries 同 dtype；m 个子空间码本，每套
      K 个 d 维码字，必须满足 D == m * d，且 m, K, d 均 > 0。
    - invlist_ids: 长度 C 的 list，第 c 项是 (n_c,) int64 张量，cell c 内
      文档的全局 id；要求 id >= 0，且同一 id 在整个倒排表中至多出现一次
      （本题不做去重）。
    - invlist_codes: 长度 C 的 list，第 c 项是 (n_c, m) int64 张量，每行是
      该文档在 m 个子空间里的码字下标，要求 0 <= code < K；同一 cell 的
      ids 与 codes 行数必须一致，n_c 可以为 0。
    - nprobe: int，默认 1；每个查询按“query 到粗中心的平方欧氏距离升序”
      选前 nprobe 个 cell，距离并列时取 cell 号小者。nprobe < 0 抛
      ValueError；nprobe > C 视为探测全部 C 个 cell；nprobe == 0 表示不探测
      任何 cell。
    - k: int，默认 10；每个查询返回的槽位数。k < 0 抛 ValueError；
      k == 0 时返回形状 (Q, 0)。

    距离口径：
    - 对查询 q、子空间 j、码字 t：
      LUT[j, t] = || q[j*d:(j+1)*d] - codebooks[j, t] ||^2。
    - 候选文档（码字行 codes[i]）的近似距离为
      sum_j LUT[j, codes[i, j]]，即 ADC 距离，查询侧不做量化。
    - 距离内部用 float64 累加（平方差逐维求和），输出时转回 queries.dtype。
      这意味着 float32 输入的输出也是 float32，允许的误差按 float32 口径判。

    返回 (ids, dists)：
    - ids: (Q, k) int64。每行先按 (距离升序, 全局 id 升序) 排列真实候选；
      真实候选不足 k 个时，尾部槽位填 -1。nprobe == 0、所选 cell 全空或
      k == 0 时该行没有真实候选（k == 0 时张量形状为 (Q, 0)）。
    - dists: (Q, k)，dtype 与 queries 相同，与 ids 逐位对齐；填充槽位为
      +inf。空候选时整行都是 +inf。
    - 两个返回值都必须是新张量，形状必须恰好是 (Q, k)。

    错误约定（以下情况一律抛 ValueError，消息内容不限）：
    - queries/centroids/codebooks 的维度不是 2/2/3；D != m*d；
      centroids 第二维不是 D。
    - 三者 dtype 不一致，或不是 float32/float64。
    - invlist_ids 或 invlist_codes 的长度不等于 C。
    - 某个 cell 的 ids 不是 1-D int64；codes 不是 (n_c, m) 的 int64；
      ids 与 codes 行数不同。
    - 任一全局 id < 0；任一码字下标不在 [0, K)。
    - nprobe < 0 或 k < 0。
    """
    raise NotImplementedError
