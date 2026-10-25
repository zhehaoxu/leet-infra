# PyramidKV: Inter-layer Budget Pyramid

## Problem Description

**PyramidKV** ([arXiv:2406.02069](https://arxiv.org/abs/2406.02069)): different Transformer layers differ enormously in their KV-cache needs — **lower layers need a longer context to recover lexical and syntactic information, while higher layers only need a few key tokens for localization**. So draw a "pyramid" along the layer index: widest at the bottom, narrowest at the top, decreasing linearly in between.

kvpress's `PyramidKVPress.get_layer_budget` produces the per-layer kept-token count with the following formulas:

- $\text{max\_cp} = W + Q(1 - r)$ ($W$=window_size, $Q$=q_len, $r$=compression_ratio)
- $\text{min\_num} = (\text{max\_cp} - W)/\beta$
- $\text{max\_num} = 2(\text{max\_cp} - W) - \text{min\_num}$
- If $\text{max\_num} \geq Q - W$: $\text{max\_num} = Q - W$; $\text{min\_num} = 2(\text{max\_cp} - W) - \text{max\_num}$
- If $Q \geq \text{max\_num} \geq \text{min\_num} \geq W$ **does not hold**: **fall back to SnapKV** with a uniform per-layer budget $\max(1, \text{round}(Q(1-r)))$
- Otherwise: $\text{steps} = (\text{max\_num} - \text{min\_num})/(\text{num\_layers} - 1)$ (steps=0 when `num_layers==1`), and layer $l$ keeps $\text{round}(\text{max\_num} - l \cdot \text{steps})$

## Rules

Implement `solve(q_len, num_layers, compression_ratio, window_size, beta)` → return a `torch.long` tensor of shape $(\text{num\_layers},)$ giving the kept count per layer.

## Example

```
q_len=100, num_layers=4, cr=0.5, W=4, β=2
max_cp = 4 + 50 = 54
min = (54-4)/2 = 25
max = (54-4)*2 - 25 = 75
No cap (75 < 96); interval is valid
steps = (75 - 25) / 3 = 16.667
budgets = [round(75), round(58.333), round(41.667), round(25)] = [75, 58, 42, 25]
```

## Constraints

- `beta >= 1`; `round` uses the Python built-in (**banker's rounding**: `round(0.5)=0`, `round(1.5)=2`)
- When `num_layers == 1`, steps=0 and it returns `[round(max_num)]` directly
- If the interval is invalid: a uniform per-layer budget `max(1, round(q_len * (1 - cr)))`
