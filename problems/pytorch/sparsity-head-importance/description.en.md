# Sparsification: Attention Head Importance Scoring and Structured Pruning

## Problem

The smallest unit of structured pruning is not an individual weight but a whole block the hardware or kernel can recognise. For multi-head attention the natural choice is a **whole head**: zeroing the head's Q/O projection slice lets the sequence-length compute be skipped entirely. The hard part is deciding **which head to drop**.

This problem scores heads with a **Wanda-style weight–activation joint saliency**: a head's weight slice is \(W_h\in\mathbb{R}^{d_\text{head}\times d}\) and its input activations over the calibration set are \(X_h\in\mathbb{R}^{N\times d_\text{head}}\), so

$$
\text{importance}_{l,h}=\|W_{l,h}\|_F\cdot\|X_{l,h}\|_F
$$

A large weight paired with near-zero activations (or the reverse) should not be kept; the product captures both the weight energy and how much this pathway is actually exercised (Michel et al. 2019 also attack structured pruning from head-level saliency). Given these scores, pruning is a single **global** sort: rank every head in every layer together and drop the lowest ones.

## Rules

Given:

- `weight`: float32, shape \((L,H,d_\text{head},D)\); slice `[l,h]` is one head;
- `calib`: float32, shape \((L,H,N,d_\text{head})\), calibration activations;
- `sparsity`: scalar \(\in[0,1]\), the fraction of **heads** to prune.

Compute:

1. head weight norm \(\|W_{l,h}\|_F\) over the whole \(d_\text{head}\times D\) slice;
2. head activation norm \(\|X_{l,h}\|_F\) over the whole \(N\times d_\text{head}\) slice;
3. `importance` = elementwise product, shape \((L,H)\);
4. number to prune \(k=\text{round}(\text{sparsity}\cdot L\cdot H)\); flatten `importance` and take the first \(k\) in **ascending** order (ties broken by flattened index \(l\cdot H+h\), ascending);
5. zero the entire \((d_\text{head},D)\) slice of each dropped head, leave the others untouched.

Return `(pruned_weight, importance)`: the first has the shape of `weight`, the second is float32 \((L,H)\). `torch.nn.utils.prune` is forbidden.

## Examples

\(L=1,H=2,d_\text{head}=2,D=2,N=2\), `sparsity=0.5`:

```
weight[0,0] = [[1,2],[2,1]]   calib[0,0] = [[3,4],[0,0]]
weight[0,1] = [[0,1],[0,0]]   calib[0,1] = [[1,0],[0,0]]

||W[0,0]||_F = sqrt(1+4+4+1) = sqrt(10) ~ 3.162278
||X[0,0]||_F = sqrt(9+16) = 5
importance[0,0] = sqrt(10)*5 ~ 15.811388

||W[0,1]||_F = 1,  ||X[0,1]||_F = 1
importance[0,1] = 1.0

k = round(0.5*1*2) = 1  -> drop the lower-scoring head 1
pruned = [[[[1,2],[2,1]], [[0,0],[0,0]]]]
```

## Constraints

- CPU float32; `importance` must use the product above, not a plain weight-magnitude comparison.
- Pruning is a **global** ranking across layers and heads, not a fixed per-layer fraction.
- A dropped head must be zeroed **as a whole slice**, not by zeroing its few smallest elements.
- `sparsity=0` must return the input bit-for-bit; `sparsity=1` must zero everything.
