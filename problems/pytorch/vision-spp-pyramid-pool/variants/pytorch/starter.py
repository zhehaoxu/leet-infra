"""SPP-Net 金字塔池化账（starter）。

口径钉死（BVLC/caffe spp_layer.cpp GetPoolingParam + pooling_layer.cpp
MaxPoolForward，SPP-Net 论文 1406.4729 的官方实现路径）：

- 每级 bin 数 k（每边），**kernel = ceil(H/k)**（H、W 各自算，可不同），
  **stride = kernel**，**pad = (kernel·k − H + 1) // 2**——remainder 对称
  分到上下（官方 ceil 整数式，不是 (remainder)//2）。
- bin (ph, pw) 的窗口边界（MaxPoolForward）：
  ```
  hs = max(0, ph·kernel − pad)      # 可为负，clip 0
  he = min(hs + kernel, H)          # 可超界，clip H
  ```
  注意 he 用 clip 前的 hs+kernel。pad>0 时相邻窗口**重叠**（与论文
  floor(i·H/k)/ceil((i+1)·H/k) 非重叠式不同——这正是官方实现账）。
- 窗口内 max（空 bin 不可能出现：题面约束 H,W ≥ max(bin_counts)）。
- 输出按 bin_counts 顺序逐级拼接 flatten（每级 k² 个值）。
- 输出 {"pooled": [...], "shapes": [[k,k]...], "kernels": [[kh,kw]...],
  "pads": [[ph,pw]...]}。
- 校验：feature 为非空规则二维有限浮点；bin_counts 非空 int ≥1；H、W
  ≥ max(bin_counts)。
"""

import math


def solve(feature, bin_counts):
    raise NotImplementedError
