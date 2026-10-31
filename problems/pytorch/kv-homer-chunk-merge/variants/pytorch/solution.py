import torch


def solve(k_left, v_left, k_right, v_right, metric, bias, reduction_mask,
          num_tokens_to_reduce, prefix_len, suffix_len):
    """HOMER 层级 chunk 合并：合并两个已处理好的兄弟 chunk，再剪到目标长度（float64 内部）。"""
    tensors = (("k_left", k_left), ("v_left", v_left), ("k_right", k_right), ("v_right", v_right))
    for name, value in tensors:
        if value.dim() != 3:
            raise ValueError(f"{name} 必须是三维 (M, H, D)")
    if k_left.dtype != k_right.dtype or k_left.dtype not in (torch.float32, torch.float64):
        raise ValueError("左右 chunk 必须同 dtype 且为 float32/float64")
    if any(value.dtype != k_left.dtype for _, value in tensors):
        raise ValueError("k/v 必须同 dtype")
    heads, head_dim = int(k_left.shape[1]), int(k_left.shape[2])
    if k_right.shape[1] != heads or k_right.shape[2] != head_dim:
        raise ValueError("左右 chunk 的 head 数与 head_dim 必须一致")
    if k_left.shape[0] != v_left.shape[0] or k_right.shape[0] != v_right.shape[0]:
        raise ValueError("k/v 的 chunk 长度必须一致")
    for name, value in (("prefix_len", prefix_len), ("suffix_len", suffix_len),
                        ("num_tokens_to_reduce", num_tokens_to_reduce)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} 必须是 >= 0 的整数")
    len_left, len_right = int(k_left.shape[0]), int(k_right.shape[0])
    if prefix_len + suffix_len < 1:
        raise ValueError("affix 长度必须大于 0（prefix_len 或 suffix_len 至少一个非零）")
    if prefix_len + suffix_len > min(len_left, len_right):
        raise ValueError("affix 长度不能超过任一 chunk 的长度")

    merged_len = len_left + len_right - prefix_len - suffix_len
    if metric.dim() != 3 or metric.shape[0] != heads or metric.shape[2] != merged_len:
        raise ValueError(f"metric 必须是 (H, Q, {merged_len})")
    if metric.dtype != k_left.dtype or bias.dtype != k_left.dtype:
        raise ValueError("metric/bias 必须与 k/v 同 dtype")
    if bias.dim() != 1 or bias.shape[0] < merged_len:
        raise ValueError(f"bias 必须是一维且长度 >= {merged_len}")
    if reduction_mask.dtype != torch.bool or tuple(reduction_mask.shape) != (merged_len,):
        raise ValueError(f"reduction_mask 必须是 ({merged_len},) 的 bool")
    if num_tokens_to_reduce >= merged_len:
        raise ValueError("num_tokens_to_reduce 必须小于合并后的长度（至少要留一个 token）")
    if num_tokens_to_reduce > int(reduction_mask.sum()):
        raise ValueError("要删的 token 数超过了可剪位置数（保护区是 +inf，永远删不掉）")

    dtype = k_left.dtype

    def merge_pair(left, right):
        left, right = left.to(torch.float64), right.to(torch.float64)
        lp, lc = left[:prefix_len], left[prefix_len:len_left - suffix_len]
        rp, rc = right[:prefix_len], right[prefix_len:len_right - suffix_len]
        tail_l, tail_r = left[len_left - suffix_len:], right[len_right - suffix_len:]
        return torch.cat([(lp + rp) * 0.5, lc, rc, (tail_l + tail_r) * 0.5], dim=0)

    merged_k, merged_v = merge_pair(k_left, k_right), merge_pair(v_left, v_right)

    weight = metric.to(torch.float64)[:, -1, :].max(dim=0).values      # 只读最后一个 query 行
    weight = weight - bias.to(torch.float64)[torch.arange(merged_len - 1, -1, -1)]
    weight = weight.masked_fill(~reduction_mask, float("inf"))          # 保护区永远选不中
    if num_tokens_to_reduce:
        removed = torch.argsort(weight, stable=True)[:num_tokens_to_reduce].sort().values
    else:
        removed = torch.empty(0, dtype=torch.int64)
    keep = torch.ones(merged_len, dtype=torch.bool)
    keep[removed] = False

    return {
        "weights": weight.to(dtype),
        "removed": removed.to(torch.int64),
        "merged_k": merged_k[keep].to(dtype).contiguous(),
        "merged_v": merged_v[keep].to(dtype).contiguous(),
    }
