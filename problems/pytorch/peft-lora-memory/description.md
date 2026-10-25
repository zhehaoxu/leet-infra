# PEFT：LoRA 可训练参数量与显存账

## 题目描述

面试常问「LoRA 到底省了多少可训练参数？省的是哪部分？」。答案：全量微调要更新每个线性层的满秩权重 \(W_0\in\mathbb{R}^{d_\text{out}\times d_\text{in}}\)，共 \(d_\text{in}\cdot d_\text{out}\) 个参数；LoRA 冻结 \(W_0\)，只训练两个低秩因子 \(A\in\mathbb{R}^{r\times d_\text{in}}\)、\(B\in\mathbb{R}^{d_\text{out}\times r}\)，共 \(r(d_\text{in}+d_\text{out})\) 个。给定一组线性层的维度与秩 \(r\)，清点两侧的**可训练参数量**并算出压缩比。

## 规则

输入 `layer_dims` 是一个列表，每个元素为二元组 \((d_\text{in}, d_\text{out})\)；`r` 为 LoRA 秩（对所有层相同）。逐层累加：

- 全量可训练参数：\(\text{full} = \sum d_\text{in}\cdot d_\text{out}\)
- LoRA 可训练参数：\(\text{lora} = \sum r\cdot(d_\text{in}+d_\text{out})\)
- 压缩比：\(\text{ratio} = \text{lora}/\text{full}\)（浮点，值域 \((0,1)\)）

返回三元组 `(full, lora, ratio)`，其中 `full`、`lora` 为整数，`ratio` 为 float。只统计低秩因子本身，不含基座 \(W_0\)、bias、优化器状态等。

## 示例

`layer_dims = [(1024,1024), (1024,4096), (4096,1024)]`、`r = 8`：

```
full = 1024*1024 + 1024*4096 + 4096*1024 = 9,437,184
lora = 8*(1024+1024) + 8*(1024+4096) + 8*(4096+1024) = 98,304
ratio = 98304 / 9437184 ≈ 0.010416667   （约 1.04%）
输出：(9437184, 98304, 0.010416666666666666)
```

## 约束

- 维度与 `r` 均为正整数；`layer_dims` 非空。
- 参数量按精确整数计，不做近似；`ratio` 用 float 除法。
- \(r\) 远小于 \(\min(d_\text{in},d_\text{out})\) 时 `ratio` 才会明显小于 1，这正是 LoRA 的省参数来源。
