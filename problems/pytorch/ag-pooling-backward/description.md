# 池化反向：max 的 argmax 路由与 adaptive 的窗口分摊

## 题目描述

池化没有可学参数，反向的全部难点都在**梯度该送回哪儿**：`max` 要把整份 \(dY\) 交给窗口内取到最大值的那一格，`avg` 要平均摊给窗口内每一格，而 `adaptive_avg` 的窗口**长度不等且可能重叠**——每个位置分到的不是 \(dY/(k_hk_w)\)，而是 \(dY/\text{该窗口的实际面积}\)。三种模式的除数与路由规则都不同，却共用同一套窗口起点公式。本题给你 `x, dY` 与模式开关，返回 `dx`。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- `mode='max'`：把 \(dY_{b,c,i,j}\) 整份给窗口内最大值位置；**并列时取行主序最小**（先上后左）的那个。
- `mode='avg'`：窗口内每个位置分得 \(dY/(k_hk_w)\)，要求 `padding=0`，否则抛 ValueError。
- `mode='adaptive_avg'`：忽略 `kernel/stride/padding`，第 \(i\) 个窗口覆盖 \([\lfloor iH/OH\rfloor,\lceil (i+1)H/OH\rceil)\)，除数是该区间的实际长度。
- 窗口越出 `x` 的部分（padding 区）不产生 `dx`；`dx` 形状恒等于 `x`；`mode` 非法抛 ValueError。

## 示例

\(4\times4\) 递增输入、`kernel=2`、`mode='max'`，数值取自 test_cases.py 的 EXAMPLE_* 常量：

```
x  = [[ 1,  2,  3,  4],    dY = [[1, 2],     四个窗口的最大值位置：
     [ 5,  6,  7,  8],           [3, 4]]     (1,1)=6, (1,3)=8, (3,1)=14, (3,3)=16
     [ 9, 10, 11, 12],
     [13, 14, 15, 16]]
dx = [[0, 0, 0, 0],
      [0, 1, 0, 2],
      [0, 0, 0, 0],
      [0, 3, 0, 4]]
```

`dx` 只有四个非零元，其余 12 格在各自窗口里都不是最大值，梯度为 0。

## 约束

- 支持 `kernel` 为 int 或 `(kh, kw)`、`stride=None`（默认等于 kernel）、`padding` 仅用于 `max`
- 与 `F.max_pool2d`/`F.avg_pool2d`/`F.adaptive_avg_pool2d` 的 `torch.autograd.grad` 对拍，float64 atol=1e-9
