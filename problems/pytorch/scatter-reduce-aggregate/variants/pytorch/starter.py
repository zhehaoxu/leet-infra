import torch


def solve(input, src, index, dim=0, reduce="sum", include_self=True):
    """scatter_reduce 聚合写入：逐点复刻 torch.Tensor.scatter_reduce_ 的 CPU 语义。

    语义：对 index 的每个坐标 c = (i_0, ..., i_{k-1})，src 在同一坐标 c 处的值 v
    会被写入 input 中「第 dim 维坐标换成 index[c]、其余维坐标不变」的位置
    （记该位置为 pos）。index 中可能出现重复坐标，同一 pos 收到多个贡献时按
    reduce 聚合；聚合按贡献到达顺序（index/src 的行主序，即同一 pos 的贡献按
    其 dim 维坐标升序排列）串行进行：

    - include_self=True：pos 以 input 原值为初值，再逐个并入 scatter 值：
      sum 逐个累加、prod 逐个累乘、amax/amin 逐个取最大/最小；
      mean 用原生 CPU 实现的增量式（Welford 型）更新——记当前均值 m、当前
      计数 n（初值 = input 原值，n = 1），每并入一个 v：n ← n+1，
      m ← m + (v − m) / n。注意这不是「总和/总数」的批量公式（浮点下二者
      舍入不同，重复元素多时可观察到明显差异）。
    - include_self=False：pos 不并入 input 原值，从 scatter 贡献的第一个值
      起算（sum/prod 从首个贡献开始累加/累乘；mean 的初值即首个贡献、n=1，
      之后按同一 Welford 更新式进行；amax/amin 只在 scatter 值中取最值）。
    - 从未被任何 index 触及的位置：两种 include_self 下都恒等于 input 原值
      （包括 mean 的 include_self=False：某位置一个 scatter 都没有时保持
      input 原值，而不是 0）。

    入参：
    - input: 浮点张量（float32 / float64），被写入的初始张量；solve 不修改它。
    - src: 与 input 同 dtype 的浮点张量，与 index 同维数。每个维度 d 上
      index.shape[d] <= src.shape[d]；src 更大时，只使用与 index 同坐标的
      那部分元素（即按坐标一一配对，多余元素不参与）。
    - index: 整型张量，与 input/src 同维数，形状与 src 满足上面的逐维 <=
      约束，且在 dim 之外各维 index.shape[d] <= input.shape[d]；dim 维本身
      允许 index.shape[dim] 大于 input.shape[dim]。取值必须落在
      [0, input.shape[dim])。
    - dim: int，默认 0，写入沿哪一维展开，支持负数索引。
    - reduce: str，"sum" / "prod" / "mean" / "amax" / "amin" 之一，
      非法取值抛 ValueError。
    - include_self: bool，默认 True。

    错误约定（除 reduce 外均与原生报错类型一致）：
    - index 出现负值或 >= input.shape[dim] 的越界值：抛 RuntimeError。
    - input/src/index 维数不一致，或形状违反上述 <= 约束，或 index 为浮点
      dtype：抛 RuntimeError。
    - dim 越界（不在 [-ndim, ndim) 内）：抛 IndexError。

    返回与 input 同形状、同 dtype 的新张量。
    """
    # TODO
    raise NotImplementedError
