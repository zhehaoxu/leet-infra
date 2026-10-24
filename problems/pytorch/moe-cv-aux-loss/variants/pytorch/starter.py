import torch


def solve(logits, k):
    """CV 辅助损失：probs=softmax; imp=probs.sum(0); load=bincount(topk(probs,k).indices);
    cv(x)=x.std(unbiased=False)/(x.mean()+1e-6); 返回 cv(imp)^2 + cv(load)^2。
    """
    # TODO
    raise NotImplementedError
