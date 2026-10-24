# 大模型拼装：Qwen2 完整因果语言模型

## 题目描述

把前面所有组件拼成**一个完整的 LLM**，端到端从 token id 走到 logits。这就是 `llm-rmsnorm`、`llm-rope-apply`、`llm-gqa-attention`、`llm-swiglu-mlp`、`llm-decoder-layer` 五题的组合应用——真实的 Qwen2-7B 只是把这里的维度与层数放大。

数据流：

$$ \text{ids} \xrightarrow{\text{embed}} X_0 \xrightarrow{\text{layer}_1}\cdots\xrightarrow{\text{layer}_L} X_L \xrightarrow{\text{final RMSNorm}} \hat X \xrightarrow{\text{lm\_head}} \text{logits} $$

## 规则

输入：

- `input_ids` $(B,S)$ 整数 token id
- `embed_weight` $(\text{vocab},\text{hidden})$：token 嵌入矩阵（`F.embedding` 就是按行查表）
- `layers`：**列表**，每个元素是一层的权重字典，键与 `llm-decoder-layer` 完全一致：`input_ln_weight, post_attention_ln_weight, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, o_weight, gate_weight, up_weight, down_weight`
- `final_norm_weight` $(\text{hidden})$：栈顶 RMSNorm 的 gain
- `lm_head_weight` $(\text{vocab},\text{hidden})$：输出投影（**无 bias**）
- `head_dim`、`base=10000.0`、`rms_eps=1e-6`

返回 `logits` $(B,S,\text{vocab})$。步骤：

1. 查表嵌入：$X = \text{embed\_weight}[\text{input\_ids}]$。
2. **RoPE 现场生成**：位置 $0..S{-}1$，$\text{inv\_freq}_i=\text{base}^{-2i/d}$，`emb=cat(freqs,freqs)`，`cos=emb.cos()`、`sin=emb.sin()`，形状 $(B,S,d)$。
3. 逐层调用 **Decoder Layer**（pre-norm → GQA 因果注意力 → 残差 → pre-norm → SwiGLU MLP → 残差），全部层共用同一 `cos/sin` 与同一因果 $-\infty$ 掩码。
4. 顶层 `RMSNorm(X, final_norm_weight)`。
5. `logits = X @ lm_head_weightᵀ`（无 bias）。

> 判题用 `transformers` 的 `Qwen2ForCausalLM`（**eager 注意力**、同权重、同 token id）作 oracle，直接对拍 logits。

## 示例

一个极小模型：`vocab=4, head_dim=4, hidden=4, intermediate=4, num_layers=2, n_q=n_kv=1`，权重由固定种子初始化，`input_ids=[[0,1,2]]` 时 logits（$3\times4$，四舍五入到 6 位）约为：

```
[[-0.070977,  0.028838, -0.070091, -0.019763],
 [-0.008341, -0.063482, -0.013246,  0.013710],
 [ 0.070258,  0.012415,  0.064580, -0.009723]]
```

（这些值来自参考实现与 oracle 的联合对拍，题面仅作形状/量级示意；判题按 `allclose` 对拍 `Qwen2ForCausalLM`。）

## 约束

- $H_k$ 整除 $H_q$；`head_dim` 为偶数；dtype float32
- 层数 = `len(layers)`，须按顺序串行、每层输出作下层输入
- `lm_head` 与 `embed` 在本合集**不共享**权重（`tie_word_embeddings=False`）
- `torch.allclose` 相对容差 $10^{-5}$ 对拍 `transformers` eager
