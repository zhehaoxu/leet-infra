import torch


def solve(x, gate_w, noise_w, noise, k, training=True):
    """中文契约 / English contract
    x [T,D], gate_w [E,D], noise_w [E,D], noise [T,E]：CPU float64；噪声为已给定的标准正态样本 /
        CPU float64; noise contains supplied standard-normal samples.
    k：真正的 Python int 且 1<=k<=E，否则 ValueError；bool 和浮点不合法 /
        a true Python int with 1<=k<=E, else ValueError; bool and floats are invalid.
    training=False 时忽略 noise 与 noise_w；不修改输入，接受非连续张量 /
        when training is False, ignore noise and noise_w; mutate nothing, accept noncontiguous tensors.
    训练时 z=x@Wg.T，z_tilde=z+noise*softplus(x@Wn.T)；按 z_tilde 降序稳定取 top-k 并只在选中项内 softmax /
        in training z=x@Wg.T and z_tilde=z+noise*softplus(x@Wn.T); stable descending top-k, softmax only over selected.
    返回 / Returns (gates [T,E], indices [T,k], effective_logits [T,E])；未选位置严格为 0 /
        gates are exactly zero at unselected positions.
    """
    raise NotImplementedError
