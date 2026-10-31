"""CornerNet corner pooling 前向账（starter）。

口径钉死（1808.01244；官方 _cpools/src/*.cpp forward 转写）：

- 四方向 running max（对 [H][W] 的每行/列）：
  * right_pool：向右扫描累积，out[i][j] = max(x[i][0..j])——每点取**左侧**
    （含自身）max；
  * left_pool：向左扫描累积，out[i][j] = max(x[i][j..W−1])——取**右侧** max；
  * top_pool：向下扫描累积，out[i][j] = max(x[i..H−1][j])——取**下方** max；
  * bottom_pool：向上扫描累积，out[i][j] = max(x[0..i][j])——取**上方** max。
  （官方文件名指扫描方向，语义上 right_pool 取左侧 max，反直觉。）
- corner 组合（官方 models/CornerNet.py：tl_pool=TopPool+LeftPool、
  br_pool=BottomPool+RightPool；tr/bl 按对称性）：
  * tl: top_pool(x) + left_pool(x)   （下方 max + 右侧 max，论文口径）
  * tr: top_pool(x) + right_pool(x)  （下方 max + 左侧 max）
  * bl: bottom_pool(x) + left_pool(x)（上方 max + 右侧 max）
  * br: bottom_pool(x) + right_pool(x)（上方 max + 左侧 max）
  两支路**各自 pool 后逐元素相加**。
- 输出 [H][W] 浮点列表；校验：x 非空矩形、全部有限浮点、mode 合法。
"""

import math


def solve(x, mode):
    """x: [H][W] 嵌套列表；mode ∈ {"tl","tr","bl","br"}。"""
    raise NotImplementedError
