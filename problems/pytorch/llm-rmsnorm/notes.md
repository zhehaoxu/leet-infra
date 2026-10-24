# RMSNorm 详解

## 为什么是大模型标配

LayerNorm 做两步：减均值（re-center）、除标准差（re-scale），还要 $\beta,\gamma$ 两个参数。
Zhang & Sennrich (2019) 发现**减均值对效果贡献很小**，于是 RMSNorm 只保留 re-scale：

$$ \text{RMS}(x)=\sqrt{\tfrac1d\sum_j x_j^2},\qquad \text{out}_i = \frac{w_i\,x_i}{\text{RMS}(x)+\epsilon\ (\text{在根号内})} $$

- 少一次求均值、少一个偏置参数 → 更快、更省显存；
- 对 token 向量的**尺度**敏感、不关心均值平移，正合 Transformer 逐位置归一化的需要。

LLaMA、Qwen、Mistral、DeepSeek 都用它，且都是 **pre-norm**（进 attn/mlp 前先归一）。

## 复刻 transformers 语义的三个坑

1. **float32 上算**：HF 先把 `hidden_states` 升到 fp32 求均方与归一化，再 `.to(input_dtype)`，
   最后才乘 `weight`。若全程用低精度或与 weight 的相乘顺序不同，边界值会有细微偏差。
2. **eps 在根号内**：`x * rsqrt(variance + eps)`，不是 `(x / rsqrt(variance)) + eps`。
3. **不减均值**：`variance` 实为**均方** `mean(x^2)`，不是 `mean((x-mean)^2)`。写成 LayerNorm 的方差就错。

## 与相邻概念

- **LayerNorm**：减均值 + 除标准差 + $\beta$，见 `layernorm-forward` / `batchnorm-forward`。
- **本合集后续**：`llm-decoder-layer` 里 pre-norm 的两个 `Qwen2RMSNorm`、`llm-qwen2-model` 末尾的 `model.norm` 都用这里的实现。
- Qwen3 的 **QK-Norm** 也是在 q/k 上套一层 RMSNorm（`llm-qk-norm`，待做）。

## 对拍口径

判题用 `transformers` 的 `Qwen2RMSNorm`（同权重、同 eps）作 oracle，`allclose` 相对容差 $10^{-5}$。
参考解手写、不调用 HF；测试另有一套独立公式（double 精度重算）交叉核对，双重保证正确。
