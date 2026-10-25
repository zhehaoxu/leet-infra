# DeepSeek MLA 多头潜在注意力 详解

## 为什么要潜在压缩（MLA 的动机）

标准 MHA/GQA 每层每 token 都要缓存 $H\times d_k$ 的 K 和 V，长上下文时 KV cache 吃掉大头显存。
DeepSeek-V2/V3 的 **MLA** 把 Key/Value 先**下投影**到一个所有头共享的低维潜在
$c_{KV}\in\mathbb{R}^{d_{kv}}$（$d_{kv}\ll H\cdot d_k$），缓存时**只存 $c_{KV}$**，用到时再经
`kv_b_proj` **上投影**展开成每头的 $k_{nope},v$。显存从 $O(H d_k)$ 降到 $O(d_{kv})$。
本 toy 里 `kv_a_proj` 把 `hidden` 压到 `kv_lora+rope`，注意力内部再展开——你写的正是这段展开。

query 侧同理可选低秩（`q_a_proj→RMSNorm→q_b_proj`），本题走 `q_lora_rank` 路径。

## 解耦 RoPE（decoupled RoPE）

潜在压缩把位置信息也压掉了会让注意力丢位置。MLA 的对策：把 RoPE 需要的分量**单独**走一条
不压缩的支路 $k_{pe}$（`kv_a_proj` 输出的后 `rope_head_dim` 维），它与压缩的 $c_{KV}$ 拼接成完整 key：
$k = [\,k_{nope}\,\|\,\tilde k_{pe}\,]$。要点：

- **RoPE 只作用在 rope 分量**（`q_rot`、`k_pe`），`nope` 内容分量不加位置——`q = [q_nope‖q̃_rot]`。
- `k_pe` 是**单头共享**的：`expand` 到 $n_h$ 个头后每个头的 rope 段完全相同，只有 `k_nope` 因头而异。
- `rope_head_dim` 必须为偶数（`rotate_half` 折半），且 HF rotary 的 `head_dim` 这里等于 `rope_head_dim`。

## 与 GQA 的关系

`llm-gqa-attention` 里 K/V 有多个独立头、靠 `repeat_kv` 把 $n_{kv}$ 组广播到 $n_q$ 头。
MLA 反过来：**只有一个共享潜在**，用 `kv_b_proj` 展开成 $n_H$ 个头的 $k_{nope}/v$，所以
`num_key_value_heads == num_attention_heads`、`num_key_value_groups=1`、**无需 repeat_kv**。
缩放仍用完整 $d_{qk}=d_{nope}+d_{rope}$ 的平方根倒数，别只除以 $d_{nope}$。

## RMSNorm 细节

两处 norm（`q_a_norm`、`kv_a_norm`）语义同 `llm-rmsnorm`：fp32 内 `x*rsqrt(mean(x²)+eps)`，
转回原 dtype 后再逐元素乘 gain，**不减均值**。`kv_a_norm` 作用在 $c_{KV}$ 的 $d_{kv}$ 维上。

## 抓错要点

- 忘记对 $c_{KV}$ 或 $q_{latent}$ 做 RMSNorm → 与 oracle 偏差大。
- 把 RoPE 错加到 `nope` 分量、或漏加到 rope 分量 → 位置相关用例全错。
- `k_pe` 忘了 `expand` 到 $n_h$ 头就 cat → 形状不匹配。
- 缩放用了 $d_{nope}$ 或 $d_v$ 而非 $d_{qk}$。

## 关联

- 上游：`llm-rmsnorm`、`llm-rope-apply`、`llm-gqa-attention`。
- 对拍：`transformers` `DeepseekV3Attention`（`attn_implementation="eager"`、`rope_interleave=False`、
  `num_key_value_heads=num_attention_heads`、`head_dim=rope_head_dim`），同权重同 cos/sin 同因果掩码，
  `allclose` $10^{-5}$。`rope_interleave=True`（HF 默认）是等价的另一种维度排布，本题固定非 interleave。
- arXiv：DeepSeek-V2 2405.04434、DeepSeek-V3 2412.19437（MLA / decoupled RoPE）。
