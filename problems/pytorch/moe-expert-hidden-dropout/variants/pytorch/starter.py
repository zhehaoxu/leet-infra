import torch


def solve(logits, x, w1, b1, w2, b2, drop_mask, k, expert_dropout, training=True):
    """中文契约 / English contract
    logits [T,E]、x [T,D]、w1 [E,D,H]、b1 [E,H]、w2 [E,H,D]、b2 [E,D]、drop_mask [T,E,H]
    全部是 CPU float64 有限张量；drop_mask 元素只能是 0.0 或 1.0（1=保留、0=丢弃）/
        all tensors are CPU float64 and finite; drop_mask entries are exactly 0.0 or 1.0.
    k 是 python int（bool 不算），1 <= k <= E；expert_dropout 是 int/float（bool 不算），0 <= p < 1；
    training 是 python bool / k and p are python scalars with the ranges above; training is a python bool.
    门控：s = softmax(logits, dim=-1)，按 s 降序（并列取专家索引小者）选 top-k，
    权重只在选中的 k 项上重新归一化 / top-k by s with ties -> smaller expert index, gates renormalized over the k picks.
    专家前向：pre = x_t @ w1[e] + b1[e]，h = relu(pre)；
    training 且 p > 0 时 h = h * drop_mask[t,e] / (1 - p)，否则 h 原样（p = 0 或 training=False 都等于关闭）/
        dropout is applied to the expert hidden activation only, after relu, with inverted scaling;
        it is disabled when training=False or p=0, and drop_mask is then ignored.
    y_e = h @ w2[e] + b2[e]，output_t = sum_j gates[t,j] * y_{expert_idx[t,j]} /
        output is the gate-weighted sum of the selected experts' outputs.
    非法输入（dtype/形状/设备/非有限/drop_mask 非 0-1/k 或 p 越界/training 非 bool）抛 ValueError；
    不修改输入，接受非连续张量 / raise ValueError on invalid input; mutate nothing; accept noncontiguous tensors.
    返回 / Returns (expert_idx int64 [T,k], gates float64 [T,k],
        hidden float64 [T,k,H]（dropout 之后的激活）, output float64 [T,D])
    """
    raise NotImplementedError
