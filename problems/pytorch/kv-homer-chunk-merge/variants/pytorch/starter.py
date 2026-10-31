import torch


def solve(k_left, v_left, k_right, v_right, metric, bias, reduction_mask,
          num_tokens_to_reduce, prefix_len, suffix_len):
    """HOMER 层级 chunk 合并（`Chunk.merge` + `_merge_power_max_last_calibrated`）的单结点算术。
    k_left/k_right:(M_L,H,D)、(M_R,H,D) 与对应 v 同 dtype（float32/float64），NHD 布局；
    prefix_len/suffix_len 为两个 chunk 共享的 affix 长度，合并后长度 M = M_L + M_R - prefix - suffix。
    weights:(M,)=metric 最后一行按 head 取 max、减去 bias 之后（bias 按「离末尾的距离」索引，
    故 bias[0] 作用在最后一个位置），reduction_mask 为 False 的位置置 +inf；
    removed:(r,) int64 升序=取最小的 r 个下标（同分取更小下标，保护区是 +inf 选不中）；
    merged_k/merged_v:(M-r,H,D)=[prefix 均值, 左 context, 右 context, suffix 均值] 按原顺序剪枝后。
    num_tokens_to_reduce=0 时不删任何 token。内部 float64，返回输入 dtype。
    返回 dict：weights / removed / merged_k / merged_v。非法输入抛 ValueError。"""
    # TODO
    raise NotImplementedError
