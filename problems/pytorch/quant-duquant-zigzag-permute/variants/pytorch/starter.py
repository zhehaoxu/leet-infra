import torch


def solve(W, block_size):
    """DuQuant 的 ZigZag 通道重排。

    输入
    ----
    W: float32 (O, H)，待重排的权重矩阵；
    block_size: int，B，块大小，要求 1 <= B <= H 且 H % B == 0。

    规则（上游 quantize/quantizer.py::permutation_zigzag）
    ------------------------------------------------
    1. 每列的键值 key_c = max_r |W[r, c]|；
    2. 把列按 (key, 原下标) 从大到小稳定排序，得到有序列表；
    3. 令 G = H / B。把有序列表按「蛇形」轮流放进 G 个桶：
       先 0,1,...,G-1，再 G-1,...,0，再 0,1,...，直到所有列放完；
    4. 每个桶内部再按 (key, 原下标) 从大到小稳定排序；
    5. 按桶 0..G-1 拼接得到 perm，permuted = W[:, perm]。

    返回
    ----
    dict:
      perm int64 (H,)：通道置换（新第 i 列来自原第 perm[i] 列）；
      permuted float32 (O, H)：重排后的权重。

    约定：键值只用于排序，内部用 float64；相等键按原下标升序（稳定排序）。
    非法输入抛 ValueError。
    """
    # TODO: 在此实现
    raise NotImplementedError
