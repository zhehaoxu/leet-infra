# 大模型组件：DeepSeek 多头潜在注意力（MLA）

## 题目描述

**MLA（Multi-head Latent Attention，DeepSeek-V2/V3）** 用**低秩潜在压缩**取代传统
KV cache：不再为每个头缓存完整的 K/V，而是把 Key/Value 压进一个所有头共享的短潜在向量
$c_{KV}$，注意力计算时再用一个上投影矩阵展开。位置编码单独走一条**解耦 RoPE（decoupled RoPE）**
支路 $k_{pe}$——它是唯一带位置的分量，且所有头共享同一份。这样 KV cache 只存
$c_{KV}$ 与 $k_{pe}$，长上下文显存大幅下降，是本合集 `llm-gqa-attention` 之后注意力家族的进阶。

数据流（对齐 HF `DeepseekV3Attention.forward` 的 `q_lora_rank` 路径、非 interleave RoPE）：

$$ q_{\text{latent}} = \text{RMSNorm}(W_{qa}\,x),\qquad q = W_{qb}\,q_{\text{latent}} \rightarrow (\text{heads},\ d_{qk}) $$
$$ [\,c_{KV}^{\text{raw}},\ k_{pe}\,] = W_{kva}\,x \rightarrow(\text{heads}=1,\ d_{kv}+d_{rope}),\qquad c_{KV}=\text{RMSNorm}(c_{KV}^{\text{raw}}) $$
$$ (\tilde q_{rot},\ \tilde k_{pe}) = \text{RoPE}(q_{rot},\ k_{pe}) \quad\text{—— 只对 rope 分量施加} $$
$$ [\,k_{nope},\ v\,] = W_{kb}\,c_{KV}\rightarrow(\text{heads},\ d_{nope}+d_v) $$
$$ q=[q_{nope}\,\|\,\tilde q_{rot}],\quad k=[k_{nope}\,\|\,\tilde k_{pe}],\quad o = \text{Attn}(q,k,v) \rightarrow W_o $$

其中 $d_{qk}=d_{nope}+d_{rope}$，注意力缩放用 $1/\sqrt{d_{qk}}$，因果 $-\infty$ 掩码、fp32 softmax。

## 规则

输入 `hidden_states` $(B,S,\text{hidden})$、已算好的 `cos`/`sin` $(B,S,d_{rope})$，以及全部权重
（线性层用 $[\text{out},\text{in}]$ 布局，`F.linear(x,W)=x@Wᵀ`）：

- `q_a_proj_weight` $(d_q,\text{hidden})$、`q_a_norm_weight` $(d_q,)$、`q_b_proj_weight` $(n_h\cdot d_{qk},\ d_q)$
- `kv_a_proj_weight` $(d_{kv}+d_{rope},\ \text{hidden})$、`kv_a_norm_weight` $(d_{kv},)$、`kv_b_proj_weight` $(n_h\cdot(d_{nope}+d_v),\ d_{kv})$
- `o_proj_weight` $(\text{hidden},\ n_h\cdot d_v)$
- 整数 `n_heads, nope_head_dim, rope_head_dim, v_head_dim`；`d_q=$` `q_a_proj_weight.shape[0]`、$d_{kv}=$ `kv_a_norm_weight.shape[0]`、`rms_eps` 默认 `1e-6`

严格按序：

1. `q_latent = RMSNorm(F.linear(hs, q_a_proj_weight), q_a_norm_weight)`；`q = F.linear(q_latent, q_b_proj_weight)` → view `(B,S,n_h,d_qk)` → `transpose(1,2)` → 沿末维 split 出 `q_nope(d_nope)`、`q_rot(d_rope)`
2. `compressed = F.linear(hs, kv_a_proj_weight)` → split 出 `c_kv_raw(d_kv)`、`k_pe(d_rope)`；`c_kv = RMSNorm(c_kv_raw, kv_a_norm_weight)`，视为单头 `(B,1,S,d_kv)`、`(B,1,S,d_rope)`
3. **仅对 `q_rot`、`k_pe` 施加 RoPE**：`x*cos.unsqueeze(1) + rotate_half(x)*sin.unsqueeze(1)`（`rotate_half` 同 `llm-rope-apply`）
4. `query = cat([q_nope, q_rot], -1)` $(B,n_h,S,d_{qk})$
5. `kv = F.linear(c_kv, kv_b_proj_weight)` → view `(B,S,n_h,d_nope+d_v)` → `transpose(1,2)` → split `k_nope(d_nope)`、`v(d_v)`；把 `k_pe` 广播到 $n_h$ 个头后 `key = cat([k_nope, k_pe], -1)`
6. `score = query @ keyᵀ * d_qk**-0.5 + 因果 -inf`，`softmax(dim=-1, dtype=fp32)`，`out = score @ v` → `transpose`→reshape `(B,S,n_h·d_v)` → `F.linear(out, o_proj_weight)`，返回 $(B,S,\text{hidden})$

> `num_key_value_heads == num_attention_heads`，即所有头共享同一份压缩潜在、**无需 repeat_kv**。
> 判题用 `transformers` 的 `DeepseekV3Attention`（**eager**、`rope_interleave=False`、同权重、同 cos/sin、同因果掩码）作 oracle。

## 示例

取 `B=1,S=3,hidden=4`、`n_heads=2, nope=2, rope=2, v=2`、`q_lora=3, kv_lora=4`，
`torch.manual_seed(11)` 后依次按上表形状 `randn` 生成 `hidden_states` 与各投影权重、
两个 norm gain 全 `1`、`cos=1, sin=0`，输出 $(B,S,4)$ 为：

```
[[[ 0.877777,  1.533879,  1.298111,  0.869258],
  [-1.622245, -1.653416, -3.259178, -2.022986],
  [-2.708383, -1.197325, -0.054776, -0.642115]]]
```

## 约束

- `rope_head_dim` 为偶数（`rotate_half` 折半）；$d_{qk}=d_{nope}+d_{rope}$，缩放用 $d_{qk}$ 而非 $d_{nope}$
- RMSNorm 在 fp32 内算 `rsqrt(mean(x²)+eps)`、转回原 dtype 再乘 gain（同 `llm-rmsnorm`）
- RoPE 只作用于 rope 分量，`nope` 内容分量**不加位置**；`k_pe` 对所有头相同
- `torch.allclose` 相对/绝对容差 $10^{-5}$ 对拍 `transformers` eager；不修改输入
