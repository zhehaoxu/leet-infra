import torch


def solve(W, H, X, bits, damp, group_size):
    """RTN 与 GPTQ 输出 MSE 及逐组误差分位数对比。

    返回 (mse_rtn, mse_gptq, rtn_quantiles, gptq_quantiles)。
    """
    # TODO: 分别实现 RTN、GPTQ，计算输出 MSE 与逐组 MAE 分位数
    raise NotImplementedError
