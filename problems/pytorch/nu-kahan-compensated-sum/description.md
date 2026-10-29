# 补偿求和与成对求和：把累加误差压回去

## 题目描述

沿 row-major 顺序把张量里所有元素加起来，返回一个 Python float。浮点加法不满足结合律：`(a+b)+c` 里 \(a+b\) 一旦被舍入，\(c\) 的低位就永久消失了，朴素累加的误差随元素个数线性增长。损失函数累加、梯度 all-reduce、BatchNorm 统计量、loss scale 之后的 fp16 求和全都在做这件事，所以「怎么加」本身就是数值稳定性的第一道考题。本题要求把四种求和格式都写出来，并让它们的输出逐位可复现——同一组数据在四种模式下会给出三个不同的整数，这正是考点。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- 一律先把元素转成 float64 再累加，不许调用 `torch.sum` 或 `math.fsum`，四种模式都要逐步手写。
- `naive`：\(s=0\) 后顺序 \(s \mathrel{+}= v\)。
- `kahan`：\(y=v-c,\ t=s+y,\ c=(t-s)-y,\ s=t\)，只返回 \(s\)，不返回补偿项 \(c\)。
- `kahan_babuska`（Neumaier）：\(t=s+v\)，若 \(|s|\ge|v|\) 则 \(c \mathrel{+}= (s-t)+v\) 否则 \(c \mathrel{+}= (v-t)+s\)，最后返回 \(s+c\)。它补上了朴素 Kahan 在「新元素比部分和还大」时把补偿整个丢掉的缺陷。
- `pairwise`：连续 `block` 个元素顺序加成一块，块与块之间递归二分归并。`block >= numel` 时逐位等于 `naive`。
- 空张量返回 0.0；只要展平后出现过 ±inf/NaN，**所有模式都退化为 naive 的顺序累加**（按 IEEE 语义传播），因为补偿项里必然出现 `inf - inf`。
- 整型/布尔张量、非法 `mode`、`block < 1` 一律抛 ValueError。

## 示例

`x = [1e16, 1.0, -1e16, 1.0]`，真值是 2（两个 1 都没被吞掉才对）：

```
x = [1e16, 1.0, -1e16, 1.0]

naive           -> 1.0
kahan           -> 1.0
kahan_babuska   -> 2.0
pairwise(b=8)   -> 1.0
pairwise(b=2)   -> 0.0
```

`1e16 + 1.0` 在 double 里就是 `1e16`（ulp(1e16) = 2），两个 1 全被抹掉；朴素 Kahan 先算 `t = 1e16 + 1` 得到 `1e16`、\(c=(t-s)-y=0\)，再加 \(-10^{16}\) 时 \(|s|=10^{16}\gg|v|=1\) 却走了错误的分支，同样把补偿丢了。Neumaier 因为比较的是 \(|s|\ge|v|\) 且返回 \(s+c\)，两个 1 都留在 \(c\) 里。`pairwise` 在 `block=8` 时整段就是一次顺序累加，所以与 naive 相同；`block=2` 时两块和 \((10^{16}+1)\approx10^{16}\) 与 \((-10^{16}+1)\approx-10^{16}\) 相加恰好抵消成 0.0，比 naive 还差。

## 约束

- 任意形状的 float32/float64 张量，空张量与非有限值都是合法输入而不是异常
- oracle 用 `math.fsum`（exhaustively carried 的精确求和）与手算整数，良态数据 float64 容差 1e-12 相对误差
- 时间 \(O(n)\)，`pairwise` 额外 \(O(n/\text{block})\) 的块和与一次二分归并
