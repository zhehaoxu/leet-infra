import torch


def solve(hidden_states, probe_w, probe_b, final_logits, temperature=1.0):
    """tuned lens 的仿射探针账：把每层隐态经各自 affine probe 映射到词表空间，
    与最终层 logits 比较，输出交叉熵与 KL 收敛账。

    参数
    ----
    hidden_states : [L, B, D]  第 0..L-1 层的残差流隐态（第 L-1 层 = 最终层之前）
    probe_w       : [L, V, D]  每层探针的线性权重
    probe_b       : [L, V]     每层探针的偏置
    final_logits  : [B, V]     最终层（LM head）logits
    temperature   : float > 0，softmax 温度（作用于双方 logits）

    返回
    ----
    dict：
        layer_logits   [L, B, V]  logits_l = probe_w[l] @ h_l + probe_b[l]
        layer_ce       [L, B]     每层 log-probs 对 target 的交叉熵，
                                   target = argmax(final_logits)（每 batch 一个）
        layer_kl       [L, B]     KL(softmax(final/T) || softmax(layer_logits/T))，
                                   逐样本求和 over V
        argmax_match   [L, B]     argmax(layer_logits) == argmax(final_logits)
        probe_rank     [L]        每个探针矩阵的秩（torch.matrix_rank）

    KL 与 CE 全部在 float64 下计算；temperature <= 0 抛 ValueError。
    """
    raise NotImplementedError
