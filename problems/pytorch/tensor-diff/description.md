# diff（n 阶差分与 prepend/append）

## 题目描述

相邻差分是时间序列特征工程、梯度一致性检查、离散化与「二阶差分判凸」里最短的一段代码：`x[1:] - x[:-1]`。`torch.diff` 把它做成一个带阶数、带边界拼接、可沿任意轴的算子——而手写实现十有八九只在 `dim=-1` 上成立，或者把输出长度算错一根。

$$\Delta^{(1)} x_i = x_{i+1} - x_i, \qquad \Delta^{(n)} x = \Delta^{(1)}\big(\Delta^{(n-1)} x\big)$$

沿 `dim` 轴先把 `prepend` 接在头部、`append` 接在尾部，得到长度 $L = \dim_x + \dim_p + \dim_a$ 的序列，再做 $n$ 次差分，输出长度 $L - n$。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- `n` 是非负 int，默认 1；`n=0` 返回**内容相同的新张量**（不是同一个对象）；负数或非 int 抛 ValueError。
- `dim` 默认 `-1`，负数按 `x.dim()` 折算；越界抛 ValueError。中间轴（`dim=0`、`dim=1`）必须与末轴同样正确。
- `prepend` / `append` 必须与 `x` 同 ndim、除 `dim` 轴外各维完全相同、dtype 相同，否则抛 ValueError。
- 差分后沿 `dim` 轴长度 $\le 0$ 时抛 ValueError（torch 此处返回空张量，本题为了判题信号统一要求非空）。

## 示例

$X$ 为 $2\times 4$（数值取自 test_cases.py 回填的 EXAMPLE_* 常量）：

```
X       = [[ 1,  4,  9, 16],
           [ 2,  3,  5,  9]]

n=1     = [[ 3,  5,  7],
           [ 1,  2,  4]]
n=2     = [[ 2,  2],
           [ 1,  2]]
dim=0   = [  1, -1, -4, -7]
prepend0= [[ 1,  3,  5,  7],     append0= [[ 3,  5,  7, -16],
           [ 2,  1,  2,  4]]              [ 1,  2,  4,  -9]]
```

`n=2` 是「差分的差分」：$5-3=2$、$7-5=2$。`prepend` 补 0 后首项变成 $1-0=1$，长度仍为 $5-1=4$；`append` 补 0 后末项变成 $0-16=-16$。平方数列的一阶差分是奇数列 $2i+1$、二阶差分恒为 2，正是这两行。

## 约束

- 输入 float32，输出 dtype 与输入一致；与 `torch.diff` 逐点对拍 atol=1e-6
- 覆盖 $n\in\{0,1,2,3,4,6\}$、`dim` 的正负写法、`prepend`/`append` 的单双侧与双侧同给
- 非法 `n`（负数、float）、越界 `dim`、`prepend`/`append` 形状或 dtype 不符、差分后长度非正，都须抛 ValueError
