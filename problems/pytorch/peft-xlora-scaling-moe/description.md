# PEFT：X-LoRA 缩放分类器驱动的 LoRA 专家混合

## 题目描述

**X-LoRA**（Mixture of LoRA Experts, arXiv:2402.07148，已入 HF PEFT `tuners/xlora`）把已训练的多个 LoRA 适配器当**冻结专家池**，只训一个缩放分类器：用基模型最后一次 hidden state 预测**逐 token × 逐层 × 逐专家**的缩放系数 \(\Lambda \in \mathbb{R}^{b\times s\times l\times n}\)，softmax 保证专家系数和为 1。推理走双前向（scaling pass 用 dummy 缩放跑基模型取 hidden states → 分类器出缩放 → 第二遍按缩放混合专家）。

官方 PEFT 实现的确定性账（锚点全部可离线重算）：

```
logits = classifier(hidden_last)                    # Linear+ReLU 链 → last Linear
scalings = logits.reshape(b, s, l, n)               # layerwise=True 时 last 输出 n·l 再 C 序重排
scalings = softmax(scalings / temperature, dim=-1)  # TemperatureScaledSoftmax
# 每层应用：
sl = scalings[:, :, layer_number, :]                # (b, s, n)
topk 路径：bool 掩码保留原值（不重归一）
softmax_topk 路径：仅非零位重归一（-inf 填充 → softmax → 零位回 0）
x_mod = x * sl[:, :, adapter].unsqueeze(-1)         # 缩放乘在进入 LoRA 的 x 上！
delta = Σ_a (x·λ_a) @ (B_a A_a)ᵀ · scaling · global_scaling_weight
```

三个关键口径：

1. **topk 是掩码不是重归一**：`top_k_lora` 路径先把非 top-k 位置零（原 top-k 值保留），`enable_softmax_topk=True` 时**再**对非零位 softmax 重归一；k=1 + softmax_topk 等价 one-hot。全零行（该 token 该层所有专家原值为 0）经 softmax_topk 必须输出全 0 而非 NaN——`masked_fill(~nonzero, 0.0)` 正是为此。
2. **缩放乘输入侧**：`apply_scalings_to_x` 把 λ 广播乘在 LoRA 专家的输入上（\((b,s,1)\times(b,s,in)\)），数学上等价于对 \(B_aA_a\) 做逐 token 列缩放：\((x\cdot\lambda)\,M^{\top} = \mathrm{einsum}(\text{bsi,bs,oi}\to\text{bso})\)。
3. **dummy 缩放（scaling pass）**：形状 \((b,s,l,n)\) 全填 `scaling_pass_value`（config 默认 0；设为 None 时语义为 \(1/n\)），供第一遍禁用 adapter 的前向使用。

## 规则

- `mode="temperature_softmax"`：logits/temperature 后 softmax(dim=-1)。
- `mode="classifier_forward"`：hidden \((b,s,h)\) + 权重链（每项 \((W,bias)\)，最后一项是 final Linear）+ `layerwise` 开关 → scalings \((b,s,l,n)\)。layerwise=False 时输出 \(n\) 类再 expand 到 l 层（各层相同）。
- `mode="dummy_scalings"`：全填 override（None → \(1/n\)）。
- `mode="maybe_topk_scalings"`：切层 + 可选 topk 掩码 + 可选 softmax_topk 重归一 → \((b,s,n)\)。
- `mode="apply_scalings_to_x"`：输入侧缩放。
- `mode="layer_forward"`：base + 多专家循环（adapter 序号 = 专家列下标），scaling 与 global_scaling_weight 只乘 LoRA 路径；scalings=None 时退化为普通 LoRA 求和（无 global weight）。
- 全程 torch float64。

## 示例

\(h=8, l=4, n=3\)（layerwise，depth 3）：

```
classifier_forward(hidden, [(W1,b1),(W2,b2),(W3,b3)], 4, 3, T=1.7) -> (2,5,4,3)
maybe_topk_scalings(scalings, layer=1, top_k=2, enable_softmax_topk=True) -> (2,5,3) 非零位和 1
layer_forward(x, scalings, [(A_a,B_a)...], scaling=2.0, gw=0.5, layer=0, W0) -> base + Σ_a 缩放路径
```

## 约束

- softmax 后每 token 每层专家系数和为 1；温度可观测（变温改变分布）。
- topk=1 + softmax_topk 输出 one-hot（最大值位 1.0）。
- `global_scaling_weight` 只乘 LoRA 增量：gw 翻倍 ⇒ 输出 = base + 2·(旧输出 − base)。
- 全零行过 softmax_topk 全 0 且有限（NaN 护栏）。
