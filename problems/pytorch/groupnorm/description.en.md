# GroupNorm

## Problem

GroupNorm is a compromise between BatchNorm and InstanceNorm: split the $C$ channels into $G$ groups and normalize with statistics computed **within each group** (across all channels of that group and all spatial positions). It is independent of the batch and is stable in small-batch settings:

$$
y = \frac{x - \mu_g}{\sqrt{\sigma_g^2 + \epsilon}}\,\gamma_c + \beta_c,\qquad
\mu_g,\sigma_g^2 = \text{statistics over the } (C/G)\times H\times W \text{ elements of group } g
$$

$G=1$ degenerates to per-sample LayerNorm, and $G=C$ degenerates to InstanceNorm.

## Rules

- Implement `solve(x, weight, bias, num_groups, eps=1e-5)`; x has shape `(N, C)` or `(N, C, *)`, and `num_groups` divides $C$.
- Hand-written approach: `reshape` to $(N, G, C/G, M)$, compute `keepdim` statistics over dimensions $(2,3)$, normalize, `reshape` back to the original shape, then multiply by $\gamma$ and add $\beta$ per channel.
- Use the **biased** variance (consistent with `F.group_norm`).
- $\gamma,\beta$ have shape $(C,)$ and are **per-channel** (different channels within a group may have different scaling).

## Examples

```
Input: x shape (1,4,1,4), channel values are [1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]
      γ=[1,1,1,1], β=[0,0,0,0], G=2
Group 0 (channels 0,1, 8 values 1..8): μ=4.5, σ²(biased)=5.25
Output channel 0: [[-1.52752376, -1.09108841, -0.65465307, -0.21821769]]
Output channel 1: [[ 0.21821769,  0.65465307,  1.09108841,  1.52752376]]
Channels 2,3 (group 1, values 9..16) have exactly the same output as channels 0,1
```

## Constraints

- float32/float64; `num_groups` ∈ [1, C] and divides C
- Results must match `torch.nn.functional.group_norm`
