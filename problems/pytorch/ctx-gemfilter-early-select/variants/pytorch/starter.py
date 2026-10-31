import torch


def solve(key_states, query_states, num_key_value_groups, topk=1024,
          kernel_size=5, norm=1.0):
    """GemFilter 早层输入裁剪的打分与选择（2409.17422，CPU 参考口径）。

    key_states：[kv_heads][n][d]，选择层的 K（RoPE 后）；
    query_states：[num_heads][q_len][d]，只取最后一个 query 位置的向量；
    num_key_value_groups：GQA 分组数，先把 K 按 repeat_kv（repeat_interleave 语义）
      复制到 num_heads 个头；
    打分：ip[h][j] = <q_last[h], K[h][j]>，对头求和得 [n]，
      再做 avg_pool1d(kernel_size, padding=kernel_size//2, stride=1)（两端零填充、
      分母恒为 kernel_size）；
    取 k_eff = min(topk, n) 个最大 pooled 值：值降序、同值按下标升序；
    norm != 1 时 distances 再除以 norm（indices 不变）。
    返回 dict：indices（top-k 下标，按上述顺序）、distances（对应 pooled/norm）、
      pooled（完整池化序列）。
    非法输入抛 ValueError。"""
    # TODO
    raise NotImplementedError
