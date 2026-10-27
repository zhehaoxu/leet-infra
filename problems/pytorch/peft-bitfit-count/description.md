# PEFT：BitFit 偏置微调参数账

## 题目描述

**BitFit**（Ben Zakee & Awadalla, ACL 2022, arXiv:2106.10199）是极简 PEFT：**只微调 bias 参数**（或其子集），其余全部冻结。BERT 系模型上它以极小的可训参数占比追平全量微调（few-shot 场景反超），实现就是"把所有 weight 的 `requires_grad` 关掉"。

工程上要回答的问题是：给定一列层，**可训参数多少、总参数多少、占比多少**。Transformer 里的"bias"按层型分三类：

- `nn.Linear(a→b)`（bias=True）：权重 \(a\cdot b\) 不可训，**bias \(b\) 个可训**——参数 \(ab+b\)、可训 \(b\)；
- `nn.LayerNorm(a)`：只有 \(\beta\)（bias 项，\(a\) 个）可训，\(\gamma\)（weight）不可训——**参数 \(2a\)、可训 \(a\)**；
- `nn.Linear(a→b)`（bias=False）：没有 bias——**参数 \(ab\)、可训 0**。

## 任务

给定层表，统计 BitFit 口径的可训参数量、总参数量与占比。

- 输入：\(\texttt{layer\_spec}\in\mathbb Z^{M\times 3}\)（int64），每行 \([\text{kind},a,b]\)：kind 0 = Linear(\(a\to b\), bias=True)、kind 1 = LayerNorm(\(a\))（此时 \(b=0\)）、kind 2 = Linear(\(a\to b\), bias=False)。
- 输出 \((\texttt{n\_trainable},\ \texttt{n\_total},\ \texttt{ratio})\)：前两者为整数，\(\texttt{ratio}=\texttt{n\_trainable}/\texttt{n\_total}\) 为浮点数。

**约定**：\(M\ge 1\)、\(1\le a,b\)；kind=1 时忽略 \(b\)；ratio 用浮点除法精确表示。容差 \(10^{-5}\)。

## 提示

- LayerNorm 的 \(\gamma\) 是 weight、不是 bias——BitFit 不训它（最常考的坑）。
- Linear 的 bias 长度是**输出维** \(b\)，不是 \(a\)。
- 全表都没有 bias 时 ratio 为 0（total 仍大于 0，别把除法写挂）。
