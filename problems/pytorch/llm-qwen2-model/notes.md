# Qwen2 完整模型 详解

## 从组件到模型

这题不引入新算子，考的是**装配顺序**——把前五块积木按 `Qwen2Model` + `lm_head` 的拓扑接起来：

```
input_ids
  → embed_tokens            (查表，F.embedding = weight[ids])
  → rotary_emb(ids, pos)    (RoPE cos/sin，全层共用、只算一次)
  → for layer in layers:    (N × DecoderLayer，串行，前一层输出即后一层输入)
  → norm                    (final RMSNorm)
  → lm_head                 (Linear hidden→vocab，无 bias)
  → logits
```

真实 Qwen2-7B 就是这套结构放大到 28 层、hidden 3584、vocab 152k。本合集用极小配置（个位数维、2–4 层）验证你理解了**拓扑**而非规模。

## 三个最容易错的装配点

1. **RoPE 只算一次、全层共用**：`rotary_emb` 在层循环**之外**调用，cos/sin 传给每一层。别在每层里重算（数值虽同，但概念上位置编码属模型级）。位置就是 `arange(S)`——prompt 阶段没有 cache，position 从 0 连续递增。
2. **embed 查表 vs 矩阵乘**：`F.embedding(ids, W)` 等价于 one-hot @ W，但直接按行索引。`ids` 是整数、`W` 是 float，别手滑做 `ids @ W`。
3. **lm_head 无 bias、且不与 embed 绑定**：HF 里很多模型 `tie_word_embeddings=True`（lm_head 复用 embed 权重），本合集显式设 `False` 并单独传 `lm_head_weight`，让数据流更清楚。

## 为什么和 oracle 能对到 maxdiff≈0

参考解从 token id 出发，逐步与 `Qwen2ForCausalLM`（eager）等价：embedding、RoPE 公式、每层 pre-norm 双残差、fp32 softmax、$-\infty$ 因果掩码都和 HF 一致。对拍时测试用**同一份从 HF 抽出的权重**同时喂给 oracle 和你的 `solve`，所以差异只可能来自实现顺序/广播/精度，而非随机初始化——这正是我们想要的判据。

## 关联

- 依赖：`llm-rmsnorm`、`llm-rope-apply`、`llm-gqa-attention`、`llm-swiglu-mlp`、`llm-decoder-layer`（本层内联复用其逻辑，使本题自包含）。
- 拓展（待做）：把这些层换成 DeepSeek 的 MLA/MoE 组件即 `llm-deepseek-model`；换 Llama 只是去掉 q/k/v 的 bias。
- 对拍：`transformers` `Qwen2ForCausalLM`（**`attn_implementation="eager"`**），同权重同 token id，`allclose` $10^{-5}$。
