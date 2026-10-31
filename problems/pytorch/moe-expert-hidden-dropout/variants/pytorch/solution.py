"""参考答案：top-k 门控重归一化 + 专家隐藏激活 dropout 的确定性前向。

三个关键口径：dropout 只施加在被选中专家的隐藏激活上（激活之后、第二个投影之前），
训练态用 inverted scaling（保留元素乘 1/(1-p)），推理态完全关闭；门控权重在 top-k
选出的专家上重新归一化。dropout mask 是显式输入，不引入任何内部随机数。
"""

import torch


def _check_tensor(name, value, dim):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    if value.dtype != torch.float64:
        raise ValueError(f"{name} must be float64")
    if value.dim() != dim:
        raise ValueError(f"{name} must be {dim}-D")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _check(logits, x, w1, b1, w2, b2, drop_mask, k, expert_dropout, training):
    for name, value in (("logits", logits), ("x", x)):
        _check_tensor(name, value, 2)
    for name, value in (("w1", w1), ("w2", w2)):
        _check_tensor(name, value, 3)
    for name, value in (("b1", b1), ("b2", b2)):
        _check_tensor(name, value, 2)
    _check_tensor("drop_mask", drop_mask, 3)

    tokens, experts = logits.shape
    if tokens < 1 or experts < 1:
        raise ValueError("logits must be a non-empty [T, E] tensor")
    if x.shape[0] != tokens or x.shape[1] < 1:
        raise ValueError("x must be [T, D] with D >= 1")
    dim = x.shape[1]
    if w1.shape[0] != experts or w1.shape[1] != dim:
        raise ValueError("w1 must be [E, D, H]")
    hidden = w1.shape[2]
    if hidden < 1:
        raise ValueError("w1 must be [E, D, H] with H >= 1")
    if b1.shape != (experts, hidden):
        raise ValueError("b1 must be [E, H]")
    if w2.shape != (experts, hidden, dim):
        raise ValueError("w2 must be [E, H, D]")
    if b2.shape != (experts, dim):
        raise ValueError("b2 must be [E, D]")
    if drop_mask.shape != (tokens, experts, hidden):
        raise ValueError("drop_mask must be [T, E, H]")
    if not bool(((drop_mask == 0.0) | (drop_mask == 1.0)).all()):
        raise ValueError("drop_mask entries must be 0.0 or 1.0")

    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError("k must be a python int")
    if not 1 <= k <= experts:
        raise ValueError("k must be in [1, E]")

    if isinstance(expert_dropout, bool) or not isinstance(expert_dropout, (int, float)):
        raise ValueError("expert_dropout must be a python int or float")
    rate = float(expert_dropout)
    if not torch.isfinite(torch.tensor(rate)) or not 0.0 <= rate < 1.0:
        raise ValueError("expert_dropout must be in [0, 1)")

    if not isinstance(training, bool):
        raise ValueError("training must be a python bool")


def solve(logits, x, w1, b1, w2, b2, drop_mask, k, expert_dropout, training=True):
    _check(logits, x, w1, b1, w2, b2, drop_mask, k, expert_dropout, training)

    shifted = logits - logits.max(dim=1, keepdim=True).values
    exp = torch.exp(shifted)
    scores = exp / exp.sum(dim=1, keepdim=True)

    expert_idx = torch.sort(scores, dim=1, descending=True, stable=True).indices[:, :k]
    raw_gates = scores.gather(1, expert_idx)
    gates = raw_gates / raw_gates.sum(dim=1, keepdim=True)

    gathered_w1 = w1[expert_idx]
    gathered_b1 = b1[expert_idx]
    pre = torch.einsum("td,tkdh->tkh", x, gathered_w1) + gathered_b1
    hidden = torch.relu(pre)

    if training and float(expert_dropout) > 0.0:
        mask = drop_mask.gather(1, expert_idx.unsqueeze(-1).expand(-1, -1, pre.shape[-1]))
        hidden = hidden * mask / (1.0 - float(expert_dropout))

    expert_out = torch.einsum("tkh,tkhd->tkd", hidden, w2[expert_idx]) + b2[expert_idx]
    output = (gates.unsqueeze(-1) * expert_out).sum(dim=1)
    return expert_idx, gates, hidden, output
