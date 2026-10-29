# Gemma2 解码层：滑窗/全局掩码与双 post-norm 装配

## 题目描述

Gemma2 的解码层在标准 pre-norm Transformer 上叠了四个自己的改动，这四件事单独看都很小，合起来却决定了整层的数值：

1. **交错掩码**：`layer_types` 让相邻层轮流走 `sliding_attention`（只看最近 `sliding_window` 个 token，含自身）和 `full_attention`（全局因果）。本题实现**一层**，由 `layer_type` 选择。
2. **attention 级 tanh softcap**：`logits = softcap · tanh(logits / softcap)`，把 logits 硬压进 \([-softcap, softcap]\)。它必须在**加掩码之前**做——掩码给的是 \(-\infty\)，若先掩码再 tanh，被掩位置会变成 \(-\text{softcap}\) 这个**有限**值，softmax 就会分到真实权重。
3. **双 post-norm**：`input_layernorm` 与 `pre_feedforward_layernorm` 是常规的 pre-norm；此外 `post_attention_layernorm` 作用在**注意力输出**上、`post_feedforward_layernorm` 作用在 **MLP 输出**上，归一化之后才加回残差。
4. **\((1+w)\) RMSNorm**：gain 初始化为 0，实际缩放是 \(1+w\)，因此"零初始化"等于恒等。

另外两个容易写错的常量：缩放用 `query_pre_attn_scalar**-0.5` 而不是 `head_dim**-0.5`（Gemma2 下两者常常相等，但语义不同）；MLP 是 **GeGLU**（`gelu_pytorch_tanh`），不是 LLaMA 的 SwiGLU。

## 规则

输入 `hidden_states` \((B,S,d)\)、已算好的 RoPE `cos`/`sin` \((B,S,d_h)\)，四个 norm 的 gain，以及 q/k/v/o 与 gate/up/down 七组无 bias 权重（线性用 \([\text{out},\text{in}]\) 布局，`F.linear(x,W)=x@Wᵀ`）。记 \(H_q\) 为 query 头数、\(H_{kv}\) 为 kv 头数、\(r=H_q/H_{kv}\)。

**RMSNorm**（四处都用同一形式）：

\[ \mathrm{RMSNorm}(x,w)=\frac{x}{\sqrt{\frac1d\sum_j x_j^2+\varepsilon}}\odot(1+w) \]

**注意力**：

1. \(h=\mathrm{RMSNorm}(x,w_{\text{in}})\)；\(Q=hW_q^\top,\ K=hW_k^\top,\ V=hW_v^\top\)，按 \((H_q,d_h)\) / \((H_{kv},d_h)\) 拆头并转置到 \((B,H,S,d_h)\)。
2. 对 \(Q,K\) 施加 RoPE：\(\;Q'=Q\odot\cos+\mathrm{rot}(Q)\odot\sin\)，\(\mathrm{rot}\) 为半维交换取负。
3. GQA 复用：第 \(h\) 个 query 头使用第 \(\lfloor h/r\rfloor\) 个 kv 头。
4. \(L=Q'K'^\top\cdot \text{query\_pre\_attn\_scalar}^{-1/2}\)。
5. **softcap**（`softcap` 非 `None` 时）：\(L\leftarrow\text{softcap}\cdot\tanh(L/\text{softcap})\)。
6. **掩码**：允许位为 \(k\le q\)；`layer_type == "sliding_attention"` 时再加 \(k>q-\text{sliding\_window}\)。其余置 \(-\infty\)，沿最后一维做 float64 softmax 得到 `attn_weights`。
7. \(A=\mathrm{attn\_weights}\cdot V\)，拼回 \((B,S,H_q d_h)\) 后过 \(o\) 投影。

**残差与 MLP**：

\[ x\leftarrow x+\mathrm{RMSNorm}(A\,W_o^\top,\ w_{\text{post-attn}}) \]

\[ x\leftarrow x+\mathrm{RMSNorm}\big(\mathrm{down}\big(\mathrm{gelu}_{\tanh}(\mathrm{gate}(h_2))\odot \mathrm{up}(h_2)\big),\ w_{\text{post-ff}}\big),\qquad h_2=\mathrm{RMSNorm}(x,w_{\text{pre-ff}}) \]

返回 `(output, attn_weights)`：`output` 形状 \((B,S,d)\)、`attn_weights` 形状 \((B,H_q,S,S)\)，都是 float32。

## 示例

取 \(B=1\)、\(S=4\)、\(d=4\)、\(H_q=H_{kv}=2\)、\(d_h=2\)、MLP 中间维 4；`hidden_states` 取单位阵（第 \(i\) 行是 \(e_i\)）、`cos=1, sin=0`（旋转退化为恒等）、四个 gain 全 0（即缩放为 1）、七个投影全为单位阵、`sliding_window=2`、`query_pre_attn_scalar=2`、`softcap=10`、`eps=1e-6`：

```
output =
[ 4.999999, 0.0,      0.0,      0.0      ]
[ 0.128880, 4.995966, 0.0,      0.0      ]
[ 0.0,      1.114066, 4.758161, 0.0      ]
[ 0.0,      0.0,      0.128880, 4.995966 ]

attn_weights[0, 0] =
[ 1.0,      0.0,      0.0, 0.0 ]
[ 0.059786, 0.940214, 0.0, 0.0 ]
[ 0.0,      0.5,      0.5, 0.0 ]
[ 0.0,      0.0,      0.5, 0.5 ]
```

读法：`hidden = e_i` 经 RMSNorm 变成 \(2e_i\)。`sliding_window=2` 让第 \(i\) 行只看到 \(k\in\{i-1,i\}\)，所以权重表每行只有两个非零。第 1 行与第 3 行的对角 logit 是 \(4/\sqrt2=2.8284\)，经 softcap 压成 \(10\tanh(0.28284)=2.7554\)，softmax 后是 \(0.940214\)（没有 softcap 时会是 \(0.9442\)）；对角伙伴的 logit 为 0，得到 \(0.059786\)。第 2 行的 head 0 看到两个 token 在 \(0,1\) 两维都是 0，logits 全 0，权重只能均分 \(0.5/0.5\)。

输出对角约 5 的来源：残差 \(e_i\) + post-norm 后的注意力 \(2e_i\) = \(3e_i\)；再做一次"RMSNorm → GeGLU → RMSNorm"恰好又贡献 \(2e_i\)（GeGLU 在 2 处的定点）。非对角项只来自注意力输出被归一化后按维度摊开。

## 约束

- `hidden_states` 为 \((B,S,d)\)、`cos`/`sin` 同形 \((B,S,d_h)\) 且 \(d_h\) 为正偶数（RoPE 需要成对维度）。
- 四个 gain 均为 \((d,)\)；`q_weight` 为 \((H_q d_h,d)\)，`k_weight`/`v_weight` 同为 \((H_{kv}d_h,d)\)；`o_weight` 为 \((d,H_qd_h)\)；`gate`/`up` 为 \((I,d)\) 同形，`down` 为 \((d,I)\)。形状自洽之外还要求 \(H_{kv}\mid H_q\)。
- `layer_type` 只接受 `"sliding_attention"` / `"full_attention"`；`sliding_window` 是 \(\ge1\) 的整数（`bool` 不算）。
- `query_pre_attn_scalar` 与 `rms_eps` 必须是有限正数；`softcap` 为 `None` 或有限正数（0 与负数抛 `ValueError`）。
- `softcap` 必须在掩码之前施加；两个 post-norm 必须作用在子层输出上；RMSNorm 必须用 \(1+w\)。
- 内部用 float64 计算，返回 float32。
