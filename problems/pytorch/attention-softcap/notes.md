# Soft-Capping 详解

## 它解决什么问题

注意力 logits 的绝对值若不受控，$\text{softmax}$ 会饱和成近似 one-hot：最大项独占概率、
其余梯度趋零，且在 bf16/fp16 下容易溢出、引发训练不稳定甚至 loss spike。硬裁剪
`clamp(-cap, cap)` 虽限制数值，但**在边界处梯度为 0**，把信息一刀切掉。

## 平滑封顶

Gemma 2 用缩放 `tanh` 做软阈值：

$$ f(x) = cap \cdot \tanh(x / cap) $$

- $|x| \ll cap$ 时 $f(x)\approx x$（几乎线性，不损信息）；
- $x \to \pm\infty$ 时 $f(x) \to \pm cap$（有界）；
- 全程光滑可导，$f'(x)=1-\tanh^2(x/cap) \in (0,1]$，边界处梯度非零。

再做普通 softmax 即可。注意 **capping 作用在 logits 上、发生在 softmax 之前**。

## 易错点

- 顺序：先 `tanh` 封顶、后 softmax；反过来（先 softmax 再 cap）完全错。
- 缩放的是**除以 cap 再乘回 cap**：`cap*tanh(x/cap)`，漏掉内层 `/cap` 会让拐点位置错。
- `cap` 极大时 $\tanh(x/cap)\approx x/cap$，退化为普通 softmax（回归用例）。
- 与温度缩放不同：temperature 是 `logits/τ` 后再 softmax（改变尖锐度）；soft-cap 是**限制上界**，二者可叠加。

## 应用

Gemma 2 全程对注意力 logits 用 `attn_logit_softcapping=50`、对输出 logits 用 `final_logit_softcapping=30`，
是其训练鲁棒性的重要配方之一，在 bf16 混合精度下尤其能避免极端 logits 导致的数值崩溃。
