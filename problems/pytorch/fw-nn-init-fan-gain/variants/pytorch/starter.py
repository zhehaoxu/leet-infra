import math

import torch


def solve(noises):
    """手写 torch.nn.init 的 fan / gain / std / bound 记账（噪声显式入参范式，
    不得调用 torch.nn.init 的任何函数）。

    参数
    ----
    noises : dict[str, Tensor]，键与场景一一对应：
      uniform 场景给 U(-1, 1) 样本（rand*2-1），normal 场景给标准正态样本。

    官方口径：
      fan_in = size(1) * prod(shape[2:])，fan_out = size(0) * prod(shape[2:])，
      dim<2 抛 ValueError；mode.lower() 须在 {fan_in, fan_out}；
      calculate_gain: linear/conv/sigmoid=1、tanh=5/3、relu=sqrt(2)、selu=3/4、
      leaky_relu=sqrt(2/(1+slope^2))（slope 缺省 0.01；param 须非 bool 的 int
      或 float）；kaiming_uniform: gain=calculate_gain(nonlinearity, a)（a 为
      负斜率、缺省 0）、std=gain/sqrt(fan)、bound=sqrt(3)*std；
      kaiming_normal: 同 std；xavier_uniform: std=gain*sqrt(2/(fi+fo))、
      bound=sqrt(3)*std；xavier_normal: 同 std；
      uniform_(a,b): 值 = a + (b-a)*(noise+1)/2；normal_(mean,std): 值 =
      mean + std*noise。

    返回
    ----
    dict：
      gains     dict[str, float]
      fans      dict[str, [fan_in, fan_out]]
      scenarios dict[str, {"dist", "param", "values"}]，param 对 uniform =
                对称半宽 bound（plain 场景 = (b-a)/2）、normal = std
    """
    raise NotImplementedError
