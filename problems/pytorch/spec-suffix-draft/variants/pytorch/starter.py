import torch


def solve(datastore, context, gamma, min_match):
    """Suffix/REST 检索草稿：在 datastore 里找 context 的最长后缀（作为某 entry 的连续子串，
    长度>=min_match），草稿取匹配紧跟其后的 <=gamma 个 token。tie-break：更长匹配 -> 更早
    entry -> 更早位置。无匹配返回空。返回 torch.long 一维。"""
    # TODO
    raise NotImplementedError
