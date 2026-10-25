import torch


def solve(mm_token_type, grid_thw):
    """简化 Qwen2-VL M-RoPE 位置编号：文本三轴=running 后 +1；连续 h×w 图像块第 k 个=(running, running+r, running+c)，块后 running+=max(h,w)。
    mm_token_type (L,) 0/1；grid_thw (M,3)。返回 int64 (3,L) 的 t/h/w 编号。"""
    # TODO
    raise NotImplementedError
