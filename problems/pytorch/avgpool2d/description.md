# 2D 平均池化

## 题目描述

Avg Pooling 对每个窗口求均值。它有一个著名的分母歧义 `count_include_pad`：

$$
\text{out}[i,j] = \frac{\sum_{\text{窗口}} x}{\text{分母}},\qquad
\text{分母} =
\begin{cases}
k_h k_w, & \text{count\_include\_pad=True}\\
\text{窗口内非 padding 像素数}, & \text{count\_include\_pad=False}
\end{cases}
$$

带 padding 时，边界窗口的有效像素更少——两种口径在边界处结果不同。

## 规则

- 实现 `solve(x, kernel_size, stride=None, padding=0, count_include_pad=True)`，x 形状 `(N, C, H, W)`。
- 输入先补 0 padding，再对每个窗口求和；除数按上表口径给。
- 禁止直接调 `F.avg_pool2d`；`unfold` 展窗求和后除法即可向量化。
- **陷阱**：`count_include_pad=False` 时每个输出位置的分母可能不一样（角上最小），用一个全局标量除数是常见错误。

## 示例

```
输入：x = 4x4 = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], kernel=2, padding=0
输出：[[[3.5, 5.5], [11.5, 13.5]]]
```

## 约束

- float32/float64；输出尺寸 floor 公式
- 结果与 `torch.nn.functional.avg_pool2d`（含 `count_include_pad` 两分支）一致
