import torch


def solve(input_ids, supervised_mask, ignore_index=-100):
    """因果 LM 移位标签：labels[b,i]=input_ids[b,i+1] 若 supervised_mask[b,i+1]>0，
    否则 ignore_index；每行末列无后继恒 ignore_index。返回 (B,L) int64。"""
    # TODO
    raise NotImplementedError
