import torch


def solve(image_feats, text_feats, logit_scale):
    """CLIP 对称 InfoNCE：li=scale*(I@T.T), lt=scale*(T@I.T), labels=arange(N)，
    loss=0.5*(CE(li,labels)+CE(lt,labels))。特征已 L2 归一 (N,D)。返回 0 维标量。禁止对比学习库。"""
    # TODO
    raise NotImplementedError
