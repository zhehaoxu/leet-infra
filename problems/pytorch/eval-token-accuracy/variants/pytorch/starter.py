import torch


def solve(pred_ids, labels, ignore_index=-100):
    """token 级准确率。pred_ids, labels 均为 int64 (B, L) 且**已逐位对齐**（本题不再移位）；
    labels[b, i] == ignore_index（默认 -100）的位置是 prompt / pad，不参与统计。

    返回二元组 (micro, macro)，都是 Python float：
      micro = 全 batch 命中数 / 全 batch 有效位数（先求和再相除）
      macro = 先对每个「有效位数 > 0」的样本求 acc，再对这些样本取算术平均；
              有效位数为 0 的样本既不进分子也不进样本数。
    整批没有任何有效位时返回 (0.0, 0.0)。"""
    # TODO
    raise NotImplementedError
