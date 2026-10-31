import torch


def solve(float_batches, quant_batches, channel_axis, init_bias=None, post_bias=True,
          channel_size=None):
    """Bias Compensation：用浮点层与量化层的输出差，流式更新一个逐通道偏置（确定性、无梯度、无随机数）。

    float_batches : float32 或 float64 (T, *R)，浮点层在 T 次校准前向上的输出
    quant_batches : 与 float_batches 同形状同 dtype，量化层在同一批输入上的输出
    channel_axis  : int 或 tuple/list of int，通道轴在**单次前向**的秩 d = dim-1 中的位置，可为负数
    init_bias     : None（等价于全零）或 (C,) 与 C = prod(bias_shape) 同元素数的张量，流式平均的初值 b_0
                    —— post_bias=False 时它进不了任何 Δ_t，第一次更新的权重又是 0，所以对结果毫无影响；
                       post_bias=True 时它被加进 Δ_0 并且系数一路保持 1，整条轨迹等于零初值轨迹平移 init_bias
    post_bias     : bool，True 表示 quant_batches 已经带上当前偏置（先减回再加），False 表示未带
    channel_size  : None 或 int，给出时用于校验 prod(bias_shape) == channel_size

    口径（内部一律 float64，返回时按输入的 dtype 给出）：
      1. 归一化通道轴：a<0 时取 a + d；越界、重复、空元组都是非法输入。
      2. bias_shape：长度 d，通道轴上填该维大小，其余轴填 1；mean_axis 按升序收集所有非通道轴。
      3. 第 t 次更新（t = 0..T-1，C 为偏置长度）：
         delta = float_batches[t] - quant_batches[t]，post_bias 时再加上当前偏置的广播形式；
         bias_updates[t] = 对 delta 沿 mean_axis 取均值后展平成长度 C 的向量；
         若 mean_axis 为空（所有轴都是通道轴），上游**跳过取均值**，直接展平 delta，本实现照抄。
      4. 流式平均：先把计数 n 自增为 t + 1，再取新偏置 = (旧偏置 * (n - 1) + bias_updates[t]) / n。
         因此第一个样本的权重是 1，之后每个样本的权重都是 1/n（与校准 batch 的大小无关）。
      5. 收尾：最终偏置按 bias_shape 广播加回 quant_batches 得到 corrected_output；
         residual_mse = mean((float_batches - corrected_output) ** 2)（对全部元素取均值）。

    返回 dict，键固定为 8 个：
      channel_axis_norm : tuple[int] 归一化后的通道轴
      bias_shape        : tuple[int] 长度 d
      mean_axis         : tuple[int] 升序的非通道轴
      bias_updates      : float (T, C) 每次更新的原始均值向量
      bias_after        : float (T, C) 每次更新之后的偏置快照
      bias              : float (C,)  最终偏置
      corrected_output  : float (T, *R) quant_batches + 最终偏置
      residual_mse      : float 0-dim  最终偏置下的输出 MSE

    非法输入抛 ValueError：T == 0、单次前向秩 d < 2、两者形状或 dtype 不同、
    channel_axis 为空/越界/重复/非整数、channel_size 与 prod(bias_shape) 不等、
    init_bias 元素数不等于 prod(bias_shape)。
    """
    # TODO
    raise NotImplementedError
