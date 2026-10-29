import torch


def solve(u, W0, apply_gs, reverse=False):
    """HRA（Householder Reflection Adaptation）：把 adapter 的方向矩阵算出来，并合并进权重。

    u        : (in_features, r) float32，可训练的反射向量组（HRA 的 hra_u）
    W0       : (out_features, in_features) float32，基座线性层权重
    apply_gs : bool，是否先对 u 做 Gram-Schmidt 正交化
    reverse  : bool，False 时为前向反射 H，True 时为反向（H.T）

    记 rank = r。HRA 的适配矩阵是 r 个 Householder 反射的乘积，写成紧凑形式：
        H = I - V @ T @ V.T
    其中 V 的列是单位化的反射向量、T 是 r x r 上三角矩阵。

    - apply_gs=True 时，V 取 u 的 reduced QR 分解的 Q，T = 2I。
    - apply_gs=False 时，V 的每一列是 u 对应列的单位化向量（逐列除以该列的 2-范数），
      并用紧凑 WY 表示还原顺序反射的交叉项：
          T_inv = 0.5 * I + triu(V.T @ V, diagonal=1)
          T     = solve_triangular(T_inv, I, upper=True)
    - reverse=True 时取 T 的转置，得到 H.T。

    合并后的权重为 W0 @ H（前向时等价于先对输入做 x @ H.T）。

    返回 (V, T, H, W_merged)：
      V        : (in_features, r) float32
      T        : (r, r) float32
      H        : (in_features, in_features) float32
      W_merged : (out_features, in_features) float32，等于 W0 @ H
    """
    # TODO
    raise NotImplementedError
