import torch


def solve(chunk_lengths, prefix_len=0, reencode_num=5, pad_to=None):
    """KVLink 的段落掩码与 link token 布局（2410.06511，CPU 参考口径）。
    chunk_lengths: 每个文档 chunk 的 token 数（>=1）；
    布局：prefix 段 [0]*prefix_len，然后每段 [j+1]*len_j + [0]*reencode_num；
    pad_to 给出时在尾部补 -1 到该长度。
    bias:(L,L) float32，0 表示可看、-inf 表示遮住：
      causal 严格上三角遮住；segment=-1 的整行整列遮住；
      目标 segment==0 可看所有合法 source；否则只能看同 segment 的 source。
    返回 dict：segment_ids:(L,) int64、bias:(L,L) float32。非法输入抛 ValueError。"""
    # TODO
    raise NotImplementedError
