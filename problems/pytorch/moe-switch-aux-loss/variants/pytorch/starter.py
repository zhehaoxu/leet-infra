import torch


def solve(logits, k):
    """Switch 负载均衡辅助损失：probs=softmax(logits,-1); sel=topk(probs,k,-1).indices;
    f_e=bincount(sel)/T; P_e=probs.mean(0); loss = E * sum(f*P)。返回标量。
    """
    # TODO
    raise NotImplementedError
