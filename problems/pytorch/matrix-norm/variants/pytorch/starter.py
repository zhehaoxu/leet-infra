import torch


def solve(x, ord="fro", dim=(-2, -1), keepdim=False):
    """matrix-norm：torch.linalg.matrix_norm 的等价实现，对 dim 选出的每根矩阵求范数。

    入参：
    - x: float32 张量，至少二维（内部 float64 计算，返回 float32）。
    - dim: 恰好两个 int，指定「行轴、列轴」，默认 (-2, -1)；负数折算，重复或越界抛 ValueError。
      被选中的两根轴之外的所有轴都当作 batch 维。
    - ord: 取以下之一，其余取值抛 ValueError：
      * "fro"（默认）：Frobenius 范数 sqrt(Σ a_ij^2)。
      * "nuc"：核范数 = 全部奇异值之和。
      * 1：诱导 1-范数 = 列绝对值之和的最大值 max_j Σ_i |a_ij|。
      * -1：min_j Σ_i |a_ij|。
      * "inf" / float("inf")：行绝对值之和的最大值 max_i Σ_j |a_ij|；-inf 取 min。
      * 2：谱范数 = 最大奇异值；-2 = 最小奇异值。
      注意矩阵的 1/inf 范数是「先沿另一轴求和、再取极值」，与向量范数的 Σ|x| 完全不同。
    - keepdim: 保留被归约的两根轴（长度 1，位置仍在原来的轴上）。

    返回形状为 batch 维（可选保留两根长度 1 的轴）的 float32 张量。
    """
    # TODO
    raise NotImplementedError
