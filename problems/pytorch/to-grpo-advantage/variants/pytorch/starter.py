import torch


def solve(rewards, group_size):
    """GRPO 组内标准化优势：view(-1,group_size) 后每组 A=(r-mean)/std(ddof=0)。
    rewards 形状 (N,) 且 N%group_size==0。返回同形状 float32。禁止现成 RL 库。"""
    # TODO
    raise NotImplementedError
