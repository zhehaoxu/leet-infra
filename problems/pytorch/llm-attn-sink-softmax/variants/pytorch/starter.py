import torch


def solve(
    scores: torch.Tensor,
    mask: torch.Tensor | None = None,
    sink_logit: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SoftMax1 / LLN：分母多一个 exp(sink_logit) 的 sink 归一。

    scores : (..., S) float32/float64，必须是有限值；要屏蔽的位置用 mask 表达
    mask : 可广播到 scores 的 bool 张量，True = 保留、False = 屏蔽（权重恒 0）；None 表示全保留
    sink_logit : 虚拟 sink 位置的 logit（StreamingLLM 的 zero token 即 0.0）

    计算口径（内部 float64）：
      1. r = max(max_{mask} scores, sink_logit)           ← sink 也参与取最大，防上溢
      2. e_j = exp(scores_j - r)（仅保留位，其余 0）；e_sink = exp(sink_logit - r)
      3. denom = sum_j e_j + e_sink
      4. weights_j = e_j / denom；sink_weight = e_sink / denom

    返回 (weights, sink_weight)：weights 与 scores 同形状、float32；
    sink_weight 形状为 scores.shape[:-1]、float32。
    每行恒有 weights.sum(-1) + sink_weight == 1，因此 sum(weights) = 1 - sink_weight < 1，
    缺口就是被 sink 吃掉的那部分概率质量。

    非法输入抛 ValueError：scores 非张量 / 非浮点 / 0 维 / 尾维为 0 / 含 nan、±inf；
    mask 非 bool 张量或不可广播；sink_logit 为 bool、非实数、nan 或 ±inf。
    """
    # TODO: 取最大值时把 sink_logit 也算进去，逐行做 sink 感知归一，再返回两部分质量
    raise NotImplementedError
