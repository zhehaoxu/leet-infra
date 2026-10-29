# 扩散采样：SEG 的查询平滑（方阵高斯模糊与均值守恒）

## 题目描述

**Smoothed Energy Guidance**（SEG）在去噪时对注意力做「能量平滑」：把若干层自注意力里的 **query 投影**替换成它的平滑版本，用这个被抹平细节的分支当作劣质模型，再按「条件减劣质」的方向做引导。平滑有两个档位——**无限模糊**（把 query 整体抹成等权平均，等价于让所有 query 变成同一个方向）和**有限高斯模糊**（在方形网格上做二维高斯卷积）。

本题只取出其中最容易被写错的一段：给定一段 query 投影，按上游的规则把它平滑成同一形状的张量。它考的是网格重排、核长取奇与夹取、反射填充、逐通道卷积这几件事的精确次序。

## 规则

输入 `query` 形状为 \((B, S, D)\)：第 0 维 batch、第 1 维 token、第 2 维通道。按 SEG 的方形图像假设，只取**前** \(n^2\) 个 token（\(n=\lfloor\sqrt{S}\rfloor\)），把它们重排成 \((B, D, n, n)\) 的二维特征图逐通道平滑，再按原顺序写回；剩余的 \(S-n^2\) 个 token **原样保留**。返回 float32 新张量，形状不变，输入不得被就地修改。

设 `sigma = blur_sigma`、`thr = blur_threshold_inf`。

**均值分支**（\(\sigma \le \text{thr}\)，注意是「不大于」就走这一支）：每个 \((b, d)\) 的特征图被它 \(n^2\) 个元素的算术平均整体替换。

**卷积分支**（\(\sigma > \text{thr}\)）：

\[ k_0 = \lceil 6\sigma\rceil + 1 - \big(\lceil 6\sigma\rceil \bmod 2\big),\qquad k = \min\big(k_0,\ n - (n \bmod 2 - 1)\big) \]

\(k_0\) 先被凑成奇数；夹取上限在 \(n\) 为偶数时是 \(n+1\)、为奇数时是 \(n\)（这样 \(k\) 仍是奇数且不超出网格尺度）。然后

\[ h = \frac{k-1}{2},\qquad x_a = -h + \frac{2h}{k-1}\,a,\qquad p_a = \frac{e^{-\frac12 (x_a/\sigma)^2}}{\sum_b e^{-\frac12 (x_b/\sigma)^2}} \]

\[ \text{out}_{b,d,i,j} = \sum_{a,c} p_a\,p_c \cdot \text{pad}_{\text{reflect}}\big(\text{patch}_{b,d}\big)_{i-\lfloor k/2\rfloor+a,\ j-\lfloor k/2\rfloor+c} \]

即一维高斯核的外积、按通道分组（depthwise）卷积，四周各反射填充 \(\lfloor k/2 \rfloor\)。\(k=1\) 时 \(x=[0]\)、核为 \([1]\)，输出与输入相同。

## 示例

\(B=1\)、\(D=2\)、\(S=10\)（\(n=3\)，前 9 个 token 参与平滑，最后一个 token 是尾巴），第 \(t\) 个 token 的值为 \([t,\ 10+t]\)：

```
sigma=1.0, thr=9999 (均值分支)
  token 0..8 = [ 4.000000, 14.000000]
  token 9    = [ 9.000000, 19.000000]      <- 尾巴原样保留

sigma=50000, thr=9999 (卷积分支，k 被夹到 3)
  token 0 = [  2.666667, 12.666667]   token 1 = [  3.000000, 13.000000]
  token 2 = [  3.333333, 13.333333]   token 3 = [  3.666667, 13.666667]
  token 4 = [  4.000000, 14.000000]   token 5 = [  4.333333, 14.333333]
  token 6 = [  4.666667, 14.666667]   token 7 = [  5.000000, 15.000000]
  token 8 = [  5.333333, 15.333333]   token 9 = [  9.000000, 19.000000]
```

## 约束

- `query` 必须是三维张量，否则抛 `ValueError`；\(S=0\) 时无方格可建，也抛 `ValueError`。
- `blur_sigma` 为有限非负数，`blur_threshold_inf` 为有限非负数；`bool`、字符串、`nan`、`inf` 一律抛 `ValueError`。
- 平滑按 \((b, d)\) 独立进行，不同通道的均值不同，不可跨通道求平均。
- 反射填充是 `reflect`（不重复边缘值），不是 `replicate`，也不是零填充。
- \(n\) 为偶数时 \(k\) 的上限是 \(n+1\)，因此 \(k\) 可以比网格边长还大，此时填充宽度为 \(\lfloor k/2\rfloor\)。
- 计算内部可用 float64，返回 float32。
