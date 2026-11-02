import torch


def solve(logits, targets=None, stride=1.0, reg_max=7, weights=None):
    # TODO: DFL（Generalized Focal Loss 的分布式框回归）。
    #   logits 形状 (N, 4, reg_max+1)：l/t/r/b 四条边各有 reg_max+1 个 bin 的原始 logit。
    #   - decode：probs = softmax(bins)，dist = Σ i·p_i · stride（bin 下标从 0 起）。
    #   - loss（给了 targets 才算）：left = int(target)、right = left+1、
    #     wl = right - target、wr = target - left，loss = wl·CE(left) + wr·CE(right)；
    #     是两条 CE 的凸组合，不是「先把 logit 插值再算一次 CE」。
    #   - targets 必须落在 [0, reg_max-1]（right 要可索引），否则 ValueError。
    #   - weights 给定时逐元素乘到 loss 上；loss_mean = loss.sum() / loss.numel()。
    #   返回 {"probs", "dist", "bin_index"}（无 targets 时）或追加
    #   {"left", "right", "w_left", "w_right", "loss", "loss_mean"}。
    raise NotImplementedError
