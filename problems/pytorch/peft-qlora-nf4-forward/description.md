# PEFT：QLoRA NF4 反量化前向与显存账

## 题目描述

**QLoRA**（Dettmers et al., NeurIPS 2023, arXiv:2305.14314）把冻结基座量化成 4-bit、训练挂在它上面的 BF16 LoRA 旁路，让 65B 模型能进单卡 48GB 微调。基座用 **NF4**（4-bit NormalFloat）码本：假设权重服从正态分布，取其分位数把 \([-1,1]\) 均分成 16 段，每段以分位点作重建值——信息论上对正态权重最优。每 **64** 个权重（同一行的连续列）共享一个 FP32 缩放（absmax），反量化为

\[W_{ij}=\text{NF4}[c_{ij}]\cdot s_{i,\lfloor j/64\rfloor}\]

NF4 码本（论文附录 E 的 4 位小数口径）：

| 码 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 值 | -1.0 | -0.6962 | -0.5251 | -0.3949 | -0.2844 | -0.1848 | -0.0911 | 0.0 | 0.0795 | 0.1609 | 0.2461 | 0.3379 | 0.4407 | 0.5626 | 0.7226 | 1.0 |

前向时反量化出 \(W\in\mathbb R^{d_\text{out}\times d_\text{in}}\)，与 LoRA 旁路加性组合（\(A\in\mathbb R^{r\times d_\text{in}}\) 下投影、\(B\in\mathbb R^{d_\text{out}\times r}\) 上投影）：

\[y=xW^{\!\top}+\frac{\alpha}{r}\,(xA^{\!\top})B^{\!\top}\]

**double quantization** 再把缩放本身压一遍：每 block 的 FP32 absmax（开销 0.5 bit/param）量化为 FP8（每 block 1 字节），每 **256** 个 block 再共享一个 FP32 二级缩放。基座存储字节数：

\[\texttt{nbytes}=\frac{n}{2}+\frac{n}{64}+\left\lceil\frac{n/64}{256}\right\rceil\cdot 4,\qquad n=d_\text{out}\cdot d_\text{in}\]

## 任务

实现 NF4 反量化前向：反量化出基座权重、加上 LoRA 旁路输出，并报告 double-quant 后基座的存储字节数。

- 输入：\(\texttt{codes}\in\{0,\dots,15\}^{d_\text{out}\times d_\text{in}}\)（\(d_\text{in}\) 是 64 的倍数）、\(\texttt{absmax}\in\mathbb R^{d_\text{out}\times(d_\text{in}/64)}\)、\(x\in\mathbb R^{N\times d_\text{in}}\)、\(A\in\mathbb R^{r\times d_\text{in}}\)、\(B\in\mathbb R^{d_\text{out}\times r}\)、\(\alpha\) 实数、\(r\) 正整数。
- 输出 \((y,\ \texttt{nbytes})\)：\(y\in\mathbb R^{N\times d_\text{out}}\)（float32），\(\texttt{nbytes}\) 为整数。

**约定**：内部用 float64 计算、输出 float32；\(\lceil\cdot\rceil\) 为整数上取整；量化（编码）方向不在本题范围。容差 \(10^{-5}\)。

## 提示

- 反量化是逐元素查表乘缩放：block 的列边界是 \(j\,\text{//}\,64\)。
- LoRA 旁路与量化基座是**加性**的：先各算各的，再相加。
- 字节数三段：权重 \(n/2\)、每 block 1 字节 FP8 缩放（\(n/64\) 个）、二级缩放每 256 block 一组、每组 4 字节 FP32。
