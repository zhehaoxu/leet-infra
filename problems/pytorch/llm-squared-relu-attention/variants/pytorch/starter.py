import torch


def solve(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    qk_norm: bool = False,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Squared-ReLU attention：relu(logits)² 替 softmax，再按行归一。

    query (..., L, hd)、key (..., S, hd)、value (..., S, vd)，前置维按广播规则对齐
    mask : 可广播到 (...broadcast..., L, S) 的 bool 张量，True = 保留、False = 置 0
    qk_norm : True 时先对 q、k 在 head 维做无权 RMSNorm（q / sqrt(mean(q²) + eps)）
    eps : qk_norm 的 RMS eps，同时是行归一化分母的下界

    计算口径（内部 float64）：
      1. qk_norm 为 True 时 q、k 各自按 head 维做 RMSNorm（无仿射）
      2. logits = q · kᵀ / sqrt(hd)
      3. a = relu(logits)²（mask 位置再置 0）
      4. denom = a.sum(-1)；weights = a / max(denom, eps)
      5. output = weights · value
    返回 (output, weights)：output 形状 (..., L, vd)、weights 形状 (..., L, S)，均 float32。

    性质提示：relu² 是二次齐次、且行归一会把任何正的常数温度精确抵消，
    所以 1/sqrt(hd) 不改变 weights；denom 为 0（整行无正 logit 或整行被 mask）时
    该行 weights 全 0、输出为 0；0 < denom < eps 时整行被等比缩小，行和 < 1。

    非法输入抛 ValueError：任一输入非张量/非浮点/维数不足/含 nan、±inf；
    q 与 k 的 head 维不一致；k 与 v 的前置维不同；前置维不可广播；
    mask 非 bool 或不可广播；qk_norm 非 bool；eps 非有限正实数。
    """
    # TODO: 可选的 q/k RMSNorm → 点积 → relu² → 行归一（分母下界 eps）→ 乘 value
    raise NotImplementedError
