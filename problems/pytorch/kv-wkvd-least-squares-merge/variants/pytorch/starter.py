import torch


def solve(V_keep, V_evict):
    """WKVD 一类做法：被驱逐的 value 投影到存活 value 张成的子空间后合并。
    V_keep:(K,d)、V_evict:(E,d) float32。求系数 C:(K,E) 使 ||V_keep^T C - V_evict^T||_F 最小
    （用伪逆，允许 V_keep 行线性相关），合并 V' = V_keep + C @ V_evict。
    返回 (merged:(K,d) float32, coef:(K,E) float32, residual:标量 float32 = Frobenius 残差)。
    d 不一致或 K/E 为 0 抛 ValueError。"""
    # TODO
    raise NotImplementedError
