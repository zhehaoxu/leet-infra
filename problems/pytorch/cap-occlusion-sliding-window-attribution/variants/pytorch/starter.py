"""Captum Occlusion 滑动窗口扰动归因账 —— 骨架。

你需要实现四个 mode，由 solve(mode, **kw) 分派：

  occlusion_grid(shape, window_shape, strides=None)
      → {"shift_counts", "num_windows", "offsets", "extents"}

  occlusion_mask(shape, window_shape, strides=None)
      → {"masks", "coverage"}

  occlusion_diff(layers, x, baseline, shape, window_shape, strides, target)
      → {"initial_eval", "window_evals", "eval_diffs", "total_attrib"}

  occlusion_attribution(layers, x, baseline, shape, window_shape, strides, target)
      → {"total_attrib", "weights", "attribution"}

关键口径：

- 所有数值走 fractions.Fraction 精确有理数，最终输出才转 float。
- 前向是显式 MLP：layers 每层 {"W": [[...]], "b": [...]}；
  层与层之间加 ReLU，最后一层不加 ReLU。
- x 可以是嵌套列表，按 shape 的形状展平后参与计算。
- window_shape 必须与 shape 同维数，且每维不得大于对应的输入维。
- strides 可以是 None（默认全 1）、一个标量整数（广播到所有维）、
  或与 shape 同维数的列表。
- 某维窗口未恰好覆盖该维时，该维 stride 不得大于窗口长度。
- baseline 可以是标量（广播）或与展平后等长的列表。
- target 采用 Python 负索引语义（-1 是最后一个输出），越界报错。

四个容易写错的点：

1. 窗口枚举是 base-conversion，**dim0 是最低位（变化最快）**。
2. 末窗会被截断：覆盖区间右端要压到 shape[d]。
3. 重叠区归因**取平均**（除以覆盖该位置的窗口数），不是直接累加。
4. 归因结果要先降级到 float32 再做除法。

具体出参格式见题面。
"""

from fractions import Fraction


def solve(mode, **kw):
    raise NotImplementedError("请实现 solve(mode, **kw)")
