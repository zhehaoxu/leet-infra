# MTP：多 token 预测的深度融合栈

## 题目描述

标准语言模型每一步只预测下一个 token。**Multi-Token Prediction**（MTP）在主干之上再挂 \(D\) 个轻量模块，让同一份主干隐状态一次性预测未来 \(D+1\) 个 token，训练信号更密、也更强调"先规划再落笔"。

DeepSeek-V3 的 MTP 结构是**串行深度栈**：第 \(k\) 个模块不只吃主干的 \(h^0\)，而是吃上一步的隐状态 \(h^{k-1}\)，再与"向前第 \(k\) 个 token 的输入嵌入"融合：

\[ h^{k}=M_k\big[\ \mathrm{RMS}(h^{k-1})\ ;\ \mathrm{RMS}(\mathrm{Emb}(x_{i+k}))\ \big] \]

两段各自归一化后拼接（维度 \(2d\)），再由 \(M_k\) 投回 \(d\) 维。\(M_k\) 之后接共享的输出头与 embedding，本题只做 \(M_k\) 这一段。

两个容易被忽略的细节：**入口是输入嵌入而不是模型自己的输出**（与主干共享同一张嵌入表），以及**每一层读到的嵌入要整体前移一位**——第 \(k\) 层用的是 \(x_{i+k}\)，不是 \(x_{i+k-1}\)。

## 规则

输入：

- `hidden`：\((B,S,d)\)，主干最后一层的隐状态，记为 \(h^0\)；
- `embeddings`：\((B,L,d)\)，输入嵌入序列；要能看到向前 \(D\) 个 token，因此 \(L\ge S+D\)；
- `W`：\((D,2d,d)\)，第 \(k\) 层的融合投影，作用在 \([\,\mathrm{RMS}(h^{k-1});\mathrm{RMS}(\text{嵌入})\,]\) 这个 \(2d\) 维拼接向量上；
- `eps`：RMSNorm 的数值稳定项，有限正实数。

递推（\(k=1,\dots,D\)）：

\[ \mathrm{RMS}(x)=\frac{x}{\sqrt{\frac1d\sum_j x_j^2+\varepsilon}}\ ,\qquad h^{k}_{b,s}=W_k^{\top}\Big[\mathrm{RMS}\big(h^{k-1}_{b,s}\big);\ \mathrm{RMS}\big(\text{embeddings}_{b,\,k+s}\big)\Big] \]

即第 \(k\) 层读的是 `embeddings[:, k : k+S]`（含左端、不含右端）。RMS 沿特征维计算，**不加可学习缩放**。

返回长度为 \(D\) 的 tuple，元素依次是 \(h^1,\dots,h^D\)，每个都是 float32、形状 \((B,S,d)\)。

## 示例

\[ h^0=\begin{bmatrix}1.0&2.0\\0.5&-1.0\end{bmatrix},\quad \mathrm{Emb}=\begin{bmatrix}0.5&0.5\\1.0&-1.0\\0.25&0.75\\-1.0&0.5\end{bmatrix} \]

（\(B=1\)、\(S=2\)、\(D=2\)、\(d=2\)，嵌入共 4 行。）取

\[ W_1=\begin{bmatrix}1&0\\0&1\\0.5&0\\0&0.5\end{bmatrix},\qquad W_2=\begin{bmatrix}0&1\\1&0\\0.5&0.5\\1&0\end{bmatrix} \]

`eps=1e-6` 时：

```
h^1 -> [[[1.132455, 0.764911], [0.856061, -0.594091]]]
h^2 -> [[[2.356817, 1.395533], [-0.806298, 0.529388]]]
```

## 约束

- 三个输入都必须是张量且维数固定：`hidden` / `embeddings` / `W` 均为 3 维，否则抛 `ValueError`。
- 形状必须自洽：`embeddings` 的 batch 与特征维同 `hidden`；`W` 为 \((D,2d,d)\) 且 \(D\ge1\)；`embeddings` 的长度至少为 \(S+D\)。
- `eps` 必须是有限正实数，`bool` 不接受。
- **结构性质**：RMSNorm 对正数放缩不敏感——`hidden` 整体乘任意正数，\(h^1\) 不变；第 \(k\) 层用到的那段嵌入整体放缩同理。第 \(k\) 层读的是 `embeddings[:, k:k+S]`：位置 0 的嵌入任何层都不会再读（\(h^0\) 已经消化过它），位置 \(S+1\) 只落在第 2 层及以后的窗口里，位置 1 落在第 1 层的窗口里。第 2 层的权重改动不影响 \(h^1\)，但会影响 \(h^2\)。
- 计算内部可用 float64，返回 float32。
