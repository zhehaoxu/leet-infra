# LLM Component: SwiGLU MLP

## Problem Description

The feed-forward network (FFN / MLP) that performs the nonlinear transformation inside each Transformer layer. Modern large models (LLaMA / Qwen / Mistral / DeepSeek) do not use the classic "two linear layers + ReLU" but **SwiGLU**, a gated variant:

$$ \text{FFN}(x) = W_{down}\,\big(\ \text{SiLU}(W_{gate}\,x)\ \odot\ (W_{up}\,x)\ \big) $$

where $\text{SiLU}(z) = z\cdot\sigma(z)$ (that is, swish) and $\odot$ is elementwise multiplication. None of the three projections carries a **bias**.

## Rules

Input $x$ (shape $(\dots, \text{hidden})$) and three linear weight tensors (all without bias, using the $[\text{out},\text{in}]$ layout of `nn.Linear.weight`): `gate_weight` and `up_weight` of shape $(\text{inter}, \text{hidden})$, `down_weight` of shape $(\text{hidden}, \text{inter})$. Return shape $(\dots, \text{hidden})$:

1. Gate branch $g = x\,W_{gate}^\top$ and $u = x\,W_{up}^\top$ (each $(\dots,\text{inter})$).
2. $\text{SiLU}(g) \odot u$.
3. Output $= (\cdot)\,W_{down}^\top$.

> The judge uses `transformers`' `Qwen2MLP` (same weights) as the oracle for cross-checking, with `hidden_act="silu"`.

## Example

```
x = [1, 2], gate_weight = up_weight = down_weight = I (2x2 identity matrix)
g = [1, 2], SiLU(g) = [1·σ(1), 2·σ(2)] = [0.7310586, 1.7615941]
SiLU(g) ⊙ u = [0.7310586·1, 1.7615941·2] = [0.7310586, 3.5231881]
output = [0.731059, 3.523188]
```

## Constraints

- $\text{inter} \ge \text{hidden}$ (real models use about $8/3\cdot\text{hidden}$; small sizes are enough for this judge)
- Weights carry no bias; the linear layers use the $[\text{out},\text{in}]$ layout of $W$ (equivalent to `F.linear(x, W)`)
- dtype float32, `torch.allclose` with relative tolerance $10^{-5}$ against `transformers`; do not modify the inputs
