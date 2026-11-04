# CPT 上下文感知提示微调

Context-aware Prompt Tuning（CPT，论文 *Context-aware Prompt Tuning: Advancing In-Context Learning with Adversarial Methods*, arXiv 2410.17222）已合入 HF PEFT（`peft/tuners/cpt/model.py` 的 `CPTEmbedding`）。CPT 结合 ICL、Prompt Tuning 与对抗优化：只微调 few-shot 上下文中的**特定 token 嵌入**（学习一个 delta），用**投影梯度下降**把 delta 限制在 per-token ε 球内防止过拟合，并用**指数衰减加权损失**缓解 recency bias（靠近末尾的示例天然占优）。

## 机制拆解

### 1. Token 类型掩码族

`cpt_tokens_type_mask` 是整数列表，语义按 `m = mask % 4` 解释：

| m | 含义 | 投影半径 |
|---|---|---|
| 1 | input-template（格式 token） | format ε |
| 2 | input（示例内容 token） | input ε |
| 3 | output-template（格式 token） | format ε |
| 0（且 mask > 0） | 示例标签组（组 id） | 冻结（1e-10） |

可训练掩码 = `m ∈ {1, 2, 3}`（`set_updated_tokens` 的反向 hook 即乘这个 0/1 mask）。标签组 `%4==0` 不更新，但其**组 id 参与 decay 损失加权**。

### 2. ε 半径与投影

两个 ε 都按模型维度归一：`ε · sqrt(token_dim / 2048)`（2048 是论文参考维度）。

`get_projection` 对 delta 权重逐 token 行做 L2 球投影：

```
token_norm = ||delta_row||_2
scale = ε_row / clamp(token_norm, min=ε_row)   # 仅对 norm > 0 的行
delta_row *= scale
```

即**只收缩不放大**：norm < ε 的行不动；norm = 0 的行严格保持为零（`projection_mask = norm > 0`）。

### 3. 前向

```
output = Embedding(frozen prompt ids) + DeltaEmbedding(projected delta)
```

prompt 嵌入由 `cpt_token_ids` 查基座词表初始化后冻结；只有 delta 可训练。

### 4. Decay 损失

对 logits/labels 做标准 next-token shift 后逐 token CE（`ignore_index=-100`），再加权平均。加权规则（`opt_weighted_loss_type == "decay"`）：

- 对 shift 后的 mask，取 `%4==0 且 >0` 的**组 id**（unique 后**倒序**遍历）；
- 第一个（最大 id）组权重 1，之后每组乘 `opt_loss_decay_factor`；
- 组 id 越小权重按几何级数衰减——论文以此抵消「靠后示例损失占比更高」的 recency bias；
- 非 -100 的位置之外完全不参与（权重与损失都只在 valid token 上平均）。

## 判题域

单入口 `solve(mode, **kw)`，五个 mode：`build_masks` / `get_epsilon` / `project` / `embed` / `decay_loss`。反向 hook、`gather_params_ctx`、词表初始化 I/O 不在域内；数值锚点 float64 域，ε 的 fp32 计算精度需锚定（官方 `torch.Tensor([...])` 默认 fp32）。
