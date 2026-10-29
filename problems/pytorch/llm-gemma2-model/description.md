# Gemma2 整模型装配：配置推导、层型交替与绑定词表头

## 题目描述

前几题分别实现了 RMSNorm、RoPE、GQA、GeGLU 和 Gemma2 解码层；这一题把它们装成一个**能直接对拍 `transformers` 完整模型**的因果语言模型。区别在于：真实推理框架拿到的是一个 config，模型必须**从配置推导出** `head_dim`、KV 头数、每层用哪种掩码，并在 `tie_word_embeddings=True` 时把词表头**绑定**到输入嵌入上。

Gemma2 在整模型层面还有四件只有装起来才看得见的事：输入嵌入乘 \(\sqrt{h}\)、偶数/奇数层交替用滑窗与全局掩码、注意力 logits 在加掩码**之前**做 tanh softcap、末端 logits 再做一次 softcap。这些都与单层实现无关，正是本题要拼对的部分。

数据流：

$$ \text{ids} \xrightarrow{\times\sqrt{h}} X_0 \xrightarrow{\text{layer}_1}\cdots\xrightarrow{\text{layer}_L} X_L \xrightarrow{\text{RMSNorm}} \hat X \xrightarrow{\text{词表头}} \text{logits} \xrightarrow{\text{softcap}} \hat{\text{logits}} $$

## 规则

- 完整签名、`config` 的全部键与缺省值只写在 starter 的 docstring 里，那是唯一判题口径。
- **维度推导**：`head_dim` 缺省为 `hidden_size // num_attention_heads`（显式给出时优先），`num_key_value_heads` 缺省等于 `num_attention_heads`，`num_hidden_layers` 缺省等于 `len(layers)`。推导值必须与权重形状吻合，否则抛 ValueError——不能静默截断，也不能悄悄用错维度继续算。
- **层型推导**：`layer_types` 缺省为「第 0 层 `sliding_attention`、第 1 层 `full_attention`、如此交替」。滑窗层额外要求 \(\text{kv} > \text{q} - \text{sliding\_window}\)（窗口含当前 token）。
- **词表头绑定**：`tie_word_embeddings=True`（缺省）时 `lm_head_weight` 必须为 `None`，词表头直接复用 `embed_weight`；为 `False` 时必须提供独立词表头。
- 每层顺序：RMSNorm（gain 按 \(1+w\)）→ q/k/v 投影 → 拆头 → RoPE → GQA 复用 k/v → \(\text{logits}=qk^{\mathsf T}/\sqrt{\text{query\_pre\_attn\_scalar}}\) → tanh softcap（可在 config 里关掉）→ 掩码置 \(-\infty\) → softmax → 乘 v → 拼头 → o_proj → **RMSNorm 后再加残差**；随后 RMSNorm → GeGLU（`gelu_tanh`）→ down → **RMSNorm 后再加残差**。
- 顶层 RMSNorm 后乘词表头；`final_logit_softcapping` 非 `None` 时做 \(\text{cap}\cdot\tanh(\text{logits}/\text{cap})\)。
- 内部 float64 计算，返回 float32 的 `(B, S, vocab)` logits。

## 示例

一个 1 层、`hidden=2`、`head_dim=2`、单头、窗口 1（只 attend 自己）、全部投影取单位矩阵、四个 norm gain 取 0（即 \(1+w=1\)）、词表头绑定的小模型：

```
embed      = [[1, 1], [1, -1], [2, 0]]
input_ids  = [[0, 1, 2]]
config     = {num_attention_heads: 1, head_dim: 2, sliding_window: 1,
              query_pre_attn_scalar: 1, tie_word_embeddings: true}

logits[0] = [2.000000, 0.000000, 2.000000]
logits[1] = [0.534581, 1.927232, 2.461812]
logits[2] = [1.414214, 1.414214, 2.828427]
```

第 0 个 token 可以手算：嵌入乘 \(\sqrt2\) 得 \([1.414,1.414]\)，RMSNorm 归一为 \([1,1]\)，注意力只看到自己于是取出自身 value，残差后为 \([2.414,2.414]\)，GeGLU 再叠加 \(1\)，最后归一为 \([1,1]\)，与绑定词表头 \([1,1]^{\mathsf T}\) 相乘得到 \([2,0,2]\)——第三列来自 `embed[2] = [2,0]`。

## 约束

- `embed_weight`、`final_norm_weight`、每层 11 个权重、独立词表头都是 float32/float64 张量且全部有限；形状必须与推导出的维度严格一致。
- `input_ids` 必须是 int32/int64 的二维张量、非空、取值在 \([0,\text{vocab})\)。
- `config` 只接受 docstring 列出的键，出现未知键抛 ValueError；`num_attention_heads` 必填。
- `head_dim` 必须为偶数；`num_attention_heads` 必须能被 `num_key_value_heads` 整除；`hidden_size` 不能被 `num_attention_heads` 整除且未显式给 `head_dim` 时抛 ValueError。
- `sliding_window >= 1` 且为整数；`query_pre_attn_scalar`、`rms_norm_eps`、`rope_theta` 为正有限实数；两个 softcap 为正有限实数或 `None`。
- 绑定/非绑定必须与是否传入 `lm_head_weight` 一致，否则抛 ValueError。
- 时间 \(O\!\left(L\cdot\left(B S h (H_q+2H_{kv})d + B H_q S^2 d + 2 B S h I\right)\right)\)，全注意力层额外占用 \(O(B H_q S^2)\) 的权重矩阵。
