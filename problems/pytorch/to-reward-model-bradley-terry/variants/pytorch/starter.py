import torch


def solve(chosen_hidden, rejected_hidden, chosen_mask, rejected_mask, weight, bias):
    """Bradley-Terry 奖励模型：每条取最后有效 token(mask=1 最大下标)的隐藏态过价值头
    (点积 weight + bias) 得 reward，loss=-logsigmoid(r_chosen-r_rejected).mean()。返回 0 维标量。"""
    # TODO
    raise NotImplementedError
