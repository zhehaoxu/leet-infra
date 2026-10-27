# 4D 加性注意力 mask 构造

## 题目描述

加性（additive）mask 是直接**加**到注意力 logits 上的 0 / 极大负数张量，是 KV-cache 解码、padding 裁剪、滑动窗口等约束的统一表达。本题构造形状 (B,1,q_len,kv_len) 的 4-D 加性 mask：**可看处恰为 0.0，不可看处恰为 \(\mathrm{finfo(dtype).min}\)**——刻意不用 \(-\infty\)，理由见规则。坐标约定：\(i\) 是查询窗内 0-based 位置，\(j\) 是 KV 缓冲内的**绝对**位置，kv_offset 是已缓存在 KV 缓冲左侧的旧 token 数，因此"当前查询行"的对角线落在列 \(j=i+\text{kv\_offset}\)。

四条可见性约束可独立开关、同时叠加，全部满足才可看：

\[j\le i+\text{kv\_offset}\ (\text{causal}),\qquad
i-(j-\text{kv\_offset})<\text{window}\ (\text{局部带}),\qquad
j<\text{kv\_len}_b,\qquad
i<\text{seq\_len}_b.\]

## 规则

- 完整签名、参数含义与默认值只写在 starter 的 docstring 里，那是唯一判题口径。四条约束逐条对应：causal=True 时叠加 \(j\le i+\text{kv\_offset}\)；给定 window 时叠加 \(i-(j-\text{kv\_offset})<\text{window}\)（window 必须 \(\ge1\)，否则 ValueError）；给定 kv_lens 时叠加 \(j<\text{kv\_len}_b\)；给定 seq_lens 时叠加 \(i<\text{seq\_len}_b\)。window 与 causal **相互独立**：causal=False 时不叠加因果约束，只留局部带（未来位置不受限）。
- **遮蔽值用 finfo.min 而非 −inf**：加性 mask 会直接加到 logits 上。若某行全部被遮（如 pad 查询行、kv_len=0 的 batch），用 −inf 会得到整行 −inf，softmax 输出 NaN；用 \(\mathrm{finfo(dtype).min}\) 则该行 logits 加完仍是有限大负数，softmax 输出近似均匀分布（\(1/L\) 摊分），训练/解码端更稳。这也是 HF transformers / flash-attn 加性 mask 的通用口径。
- **batch 广播语义**：B 取各给定 lens 序列长度的最大值，都缺省时 B=1；每个 lens 序列长度须为 B 或 1——长度 1 沿 batch 复制（例如查询行 lens 长度 1 而 KV 列 lens 长度 B 时，查询行约束逐 batch 复制、KV 列约束逐 batch 独立）；两者都给且长度都 >1 时必须相等，否则 ValueError。
- **lens 语义**：查询行 lens 各项 \(\le\) q_len、KV 列 lens 各项 \(\le\) kv_len（越界、负数、非 int 序列抛 ValueError）；\(i\ge\text{seq\_len}_b\) 的 pad 查询行**整行**不可看，\(j\ge\text{kv\_len}_b\) 的 pad KV 列**整列**不可看。
- 返回 (B,1,q_len,kv_len)、给定 dtype 的连续张量：可看处**恰为** 0.0、不可看处**恰为** \(\mathrm{finfo(dtype).min}\)；q_len 或 kv_len 为 0 时返回尾维为空的合法张量。dtype 只接受 float32/float64。q_len/kv_len/kv_offset 必须是非负 int（负数或非整数抛 ValueError）。全程不使用随机数。

## 示例

q_len=3、kv_len=5、kv_offset=2、causal=True（1=可看 0=被遮，batch 0）：

\[\begin{bmatrix}1&1&1&0&0\\1&1&1&1&0\\1&1&1&1&1\end{bmatrix}\]

第 \(i\) 行的可看集恰为 \(j\le i+2\)：行 0 看到 j∈{0,1,2}，行 1 到 j=3，行 2 到 j=4——对角线 \(j=i+2\) 及其左侧全部可见。被遮位置的具体值是 \(-3.4028235\times10^{38}\)（float32 的 finfo.min）。

窗口组合示例（kv_offset=1、causal=True、window=1）：每行只剩对角线一点，可看集为 \(j=i+1\)：

\[\begin{bmatrix}0&1&0&0&0\\0&0&1&0&0\\0&0&0&1&0\end{bmatrix}\]

## 约束

- q_len/kv_len/kv_offset 非负 int、window \(\ge1\)、dtype ∈ {float32, float64}、lens 各项非负且不越界；违规一律 ValueError
- mask 只允许两种取值：0.0 与 \(\mathrm{finfo(dtype).min}\)（fp32 ≈ \(-3.4028\times10^{38}\)、fp64 ≈ \(-1.7977\times10^{308}\)），判题逐位 torch.equal
- 判别口径直接断言：causal-only 且 kv_offset=0 时下三角结构与 torch.tril 逐位一致；kv_offset=3 时对角带整体右移 3 列（与 kv_offset=0 显著不同）；window 足够大时逐位退化为纯因果；lens 全额（KV 列 lens=[kv_len]、查询行 lens=[q_len]）时与不给 lens 逐位相同；pad 查询行整行、pad KV 列整列 finfo.min；KV 列 lens=[0,3] 时 batch 0 整 batch 全遮——这正是"全遮行 softmax 只均匀摊分、不出 NaN"的机制所在
