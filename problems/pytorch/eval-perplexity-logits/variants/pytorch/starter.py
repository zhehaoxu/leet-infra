import torch


def solve(logits, labels, ignore_index=-100):
    """语言模型困惑度（perplexity）。输入 logits (B,L,V) float32＝模型在每个位置对「下一个词」
    的未归一化打分；labels (B,L) int64＝与输入同长的目标 token id，其中 prompt/pad 位置写作
    ignore_index（默认 -100）。

    右移对齐：位置 i 的 logits 预测 labels[i+1]，故配对为 logits[:, :L-1, :] 与 labels[:, 1:]。
    逐位 NLL = -log_softmax(logits)[..., label_id]，仅统计 label != ignore_index 的位置；
    ppl = exp( NLL 之和 / 有效位置数 )。有效位置数为 0 时返回哨兵 -1.0（空评测集无定义）。
    返回 0 维 float32 张量。"""
    # TODO
    raise NotImplementedError
