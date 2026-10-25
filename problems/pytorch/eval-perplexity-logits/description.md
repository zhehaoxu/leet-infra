# 评测：从 logits 计算困惑度 PPL

## 题目描述

困惑度（perplexity, PPL）是语言模型最基础的 intrinsic 指标：把模型在每个位置分配给**真实下一词**的概率取几何平均再倒数，等价于 \(\exp(\overline{\text{NLL}})\)。它的可读性来自「等效分支数」——\(\text{PPL}=1\) 表示模型毫不迟疑，\(\text{PPL}=V\)（词表大小）表示与在词表上均匀瞎猜没有区别。Mikolov 的 RNNLM 报告（2012）与 HuggingFace `evaluate` 的 `perplexity` 都是这个口径。

## 规则

- `logits`：float32，形状 \((B, L, V)\)——模型对每个输入位置「下一个 token」的未归一化打分。
- `labels`：int64，形状 \((B, L)\)——与输入等长的目标 token id；prompt 段与 pad 段这些不参与监督的位置一律写成 `ignore_index`（默认 \(-100\)）。
- **右移对齐**：位置 \(i\) 的 logits 用来预测第 \(i+1\) 个标签，所以配对集合是前 \(L-1\) 个位置的 logits 对上后 \(L-1\) 个标签；最后一个位置的 logits 没有目标可配，直接丢弃；`labels` 的第 \(0\) 位不对应任何预测。
- 逐位损失：\(\text{nll}_i = -\log\mathrm{softmax}(\text{logits}_i)[\text{label}_i]\)，其中 log-softmax 必须数值稳定（先减行最大值）。
- 分母**只有有效位数**：统计满足 \(\text{label}\neq\) `ignore_index` 的配对个数 \(n\)，PPL \(=\exp\big(\frac{1}{n}\sum \text{nll}\big)\)——即「所有有效 token 的 micro 平均」，不允许先按样本平均再对样本平均。
- 有效位数为 \(0\)（纯 prompt、全 pad、或 \(L=1\) 使右移后无配对）时返回哨兵 **\(-1.0\)**：非空评测集上 PPL 恒 \(\ge 1\)，\(-1\) 不会被真实值撞到。
- 返回 **0 维 float32 张量**。禁止调用现成评测库。

## 示例

```
logits[0] = [[ 2.0,  0.5, -1.0],
             [ 0.0,  1.0,  3.0],     labels[0] = [-100, 0, 2]
             [ 0.2,  0.1,  0.3]]     shift 后参与配对的标签 = [0, 2]
                                     （labels[0][0] = -100 是 prompt，右移后已不在窗口内；
                                       位置 2 的 logits 没有下一个标签，丢弃）

logits[1] = [[ 1.0,  1.0,  1.0],
             [-2.0,  0.5,  0.5],     labels[1] = [1, -100, 1]
             [ 0.0,  0.0,  4.0]]     shift 后参与配对的标签 = [-100, 1]
                                     （只有 1 个有效位）

有效配对 3 个：
(行0, 位置0 -> id0) nll = 0.241311
(行0, 位置1 -> id2) nll = 0.169846
(行1, 位置1 -> id1) nll = 0.733370
mean_nll = 1.144527 / 3 = 0.381509
ppl = exp(0.381509) = 1.4644929
```

## 约束

- CPU float32；\(V\ge 2\)，\(B, L \ge 1\)；有效 label 满足 \(0 \le \text{id} < V\)。
- 非空有效集上 \(\text{PPL} \in [1, +\infty)\)。
- **退化性质**：某行 logits 全相等（softmax 均匀）且所有有效位都如此 \(\Rightarrow \text{PPL}=V\)；每个有效位都把概率压到正确 id \(\Rightarrow \text{PPL}\to 1\)；`labels` 全是 `ignore_index` 或 \(L=1\) \(\Rightarrow\) 返回 \(-1.0\)。
- 只统计 shift 之后的位置：把「不移位」或「位移反方向」写错，样例里的三位 nll 立刻对不上。
