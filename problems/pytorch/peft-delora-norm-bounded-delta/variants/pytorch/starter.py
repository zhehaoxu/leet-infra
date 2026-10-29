import torch


def solve(W0, A, B, lam, w_norm):
    """DeLoRA：按每行的范数把低秩更新归一化，得到一个有界的方向矩阵。

    W0     : (out_features, in_features) float32，基座权重
    A      : (r, in_features) float32，低秩因子 A
    B      : (out_features, r) float32，低秩因子 B
    lam    : float，边界参数 delora_lambda（>= 0）
    w_norm : (in_features,) float32，基座权重每列的范数（输入通道尺度）

    记 rank = r、An_i = max(||A_i||, 1e-4)（A 的第 i 行范数）、Bn_i = max(||B^i||, 1e-4)
    （B 的第 i 列范数）。每个秩 1 分量的缩放为

        scaling_i = lam / r / (An_i * Bn_i)

    更新矩阵与合并权重（乘上输入通道尺度 w_norm）：

        delta  = B @ diag(scaling) @ A
        delta  = delta * w_norm        # 按列（输入通道）缩放
        merged = W0 + delta

    注意缩放只依赖 A 的行范数与 B 的列范数，与数据无关；w_norm 是训练开始时就冻结的
    基座列范数，不随训练更新。

    返回 (scaling, delta, merged)：
      scaling : (r,) float32
      delta   : (out_features, in_features) float32
      merged  : (out_features, in_features) float32
    """
    # TODO
    raise NotImplementedError
