# 自适应平均池化

## 题目描述

Adaptive Avg Pooling 不指定窗口大小，而是**指定输出尺寸**，由输入/输出尺寸反推每个输出位的采样区间。PyTorch 的边界公式：

$$
\text{out}[i] = \text{mean}\Big(x\big[\, \big\lfloor \tfrac{i \cdot H}{o_h} \big\rfloor \;:\; \big\lceil \tfrac{(i+1)\cdot H}{o_h} \big\rceil \,\big] \Big)
$$

区间长度不均等、还可能重叠（如 $H=5, o_h=4$ 时列为 $[0,2),[1,3),[2,4),[3,5)$）。

## 规则

- 实现 `solve(x, output_size)`，x 形状 `(N, C, H, W)`，`output_size` 为 int 或 `(oh, ow)`。
- 高和宽各自套上面的 floor/ceil 公式；对每个 $(i,j)$ 区间求均值。
- 禁止调 `F.adaptive_avg_pool2d`；对输出位小循环 + 切片 `.mean()` 是完全合法的写法。
- **陷阱**：用「固定窗宽 $H/o_h$ 取整」之类的朴素切分在非整除尺寸下必错，必须 start 用 floor、end 用 ceil。

## 示例

```
输入：x = 4x4 = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], output_size=2
输出：[[[3.5, 5.5], [11.5, 13.5]]]    # 4->2 整除，区间恰好是两行两列的等宽块
```

## 约束

- float32/float64；`output_size` 可以大于输入（此时多数区间长度为 1）
- 结果与 `torch.nn.functional.adaptive_avg_pool2d` 一致
