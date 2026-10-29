import torch


def solve(x, gamma, beta, dY, eps=1e-5):
    """BatchNorm（特征维 C、按 batch 轴归一）训练态反向：返回 (dX, dGamma, dBeta)。

    前向：mu、var 沿**第 0 轴**（batch）逐特征计算，方差是有偏估计（unbiased=False），
    xhat = (x − mu) / sqrt(var + eps)，y = gamma · xhat + beta。

    入参：
    - x: (N, C) 二维，N 是 batch 大小；dY 与 x 同形；gamma/beta 是 (C,)。
    - eps: float ≥ 0，负值抛 ValueError；形状不符抛 ValueError。

    口径（务必与 autograd 逐位一致）：
    - 记 a = dY · gamma，则
      dX = ( a − mean_N(a) − xhat · mean_N(a · xhat) ) / std，三个 mean 都沿 batch 轴。
      这三项分别来自「直通」「mu 也由 x 决定」「var 也由 x 决定」，其中后两项各含
      1/N 因子；把它们当成 1/N 与 1/N² 混着写就是经典的「双重计入」错误。
    - dGamma = Σ_N dY · xhat，dBeta = Σ_N dY，形状 (C,)。
    - 用同一批数据自身统计量归一（训练态），**不要**用 running_mean/running_var。
    - N=1 时 mu=x、var=0，公式自然退化为 dX=0；eps=0 时 dX 沿 batch 的两个投影
      ΣdX = 0 与 Σ dX·xhat = 0 精确成立，可作自检。
    - 返回 dtype 与 x 一致（内部 float64 累计）。
    """
    # TODO
    raise NotImplementedError
