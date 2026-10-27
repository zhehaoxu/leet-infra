# PEFT：Prefix-Tuning 虚拟 KV 前缀注意力

## 题目描述

**Prefix-tuning**（Li & Liang, ACL 2021, arXiv:2103.10385）在注意力的 KV 侧拼上一段**可学习的虚拟前缀**：不改动任何基座参数，只训练 \(L\) 个虚拟 token 的 \(K,V\) 向量，让真实 token 的 query 去"看"这段前缀：

\[\text{attn}=\text{softmax}\!\Bigl(\frac{Q\,[P_k;K]^{\!\top}}{\sqrt d}\Bigr)[P_v;V]\]

其中 \([P_k;K]\)、\([P_v;V]\) 是**前缀在前**的拼接：\(P_k,P_v\in\mathbb R^{L\times d}\)，\(K,V\in\mathbb R^{S\times d}\)，\(Q\in\mathbb R^{N_q\times d}\)。

可训参数有两种实现（PEFT `PrefixEncoder` 两形态，中间隐藏维 \(h=d\)）：

- **直接形态**：\(P_k,P_v\) 合成一张 embedding 表——\(n_\text{direct}=2Ld\)。
- **重参数形态**（论文推荐，MLP 稳定优化）：先训 \(L\times d\) 的 embedding，再过 \(\text{Linear}(d\to h)\to\tanh\to\text{Linear}(h\to 2Ld)\) 生成 KV：

\[n_\text{reparam}=\underbrace{Ld}_{\text{embedding}}+\underbrace{(dh+h)}_{\text{第一层 Linear}}+\underbrace{(h\cdot 2Ld+2Ld)}_{\text{第二层 Linear}}\]

## 任务

实现带虚拟前缀的单头注意力前向，并报告两种形态的可训参数量。

- 输入：\(P_k,P_v\in\mathbb R^{L\times d}\)（\(L\ge 0\)），\(K,V\in\mathbb R^{S\times d}\)，\(Q\in\mathbb R^{N_q\times d}\)。
- 输出 \((\texttt{attn},\ \texttt{n\_direct},\ \texttt{n\_reparam})\)：\(\texttt{attn}\in\mathbb R^{N_q\times d}\)（float32），两个参数量为整数。

**约定**：内部用 float64 计算、输出 float32；softmax 在分数最后一维上做；\(h=d\)；不使用任何现成 PEFT/transformers 库。容差 \(10^{-5}\)。

## 提示

- 拼接顺序：前缀在前、真实 KV 在后——顺序错了参数量对、数值错。
- \(L=0\) 时应退化为标准注意力 \(\text{softmax}(QK^\top/\sqrt d)V\)（正好当自检）。
- 重参数形态的 MLP 只数权重+偏置，Tanh 无参数；它的参数比直接形态**多**，收益是优化稳定性。
