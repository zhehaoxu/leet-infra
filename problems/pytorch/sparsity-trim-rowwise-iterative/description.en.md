# Sparsification: TRIM Row-Wise Sparsity via Output-Cosine Feedback

## Problem

A uniform per-row sparsity \(s\) prunes the same fraction of columns from every output row, yet rows differ in how much they suffer from those deletions: the same \(s\) leaves one row's direction almost intact and visibly rotates another. TRIM (*Achieving Extreme Sparsity with Targeted Row-wise Iterative Metric-driven Pruning*, 2505.16743) therefore gives **each output row its own sparsity** and adjusts it iteratively using the cosine similarity between the unpruned and pruned row outputs as feedback.

### 1. Ordering comes from the Wanda metric

First compute element scores

\[
m_{ij}=|W_{ij}|\cdot\sqrt{\mathbb{E}_t[X_{t,j}^2]}
\]

and stable-sort each row ascending to get `sorted_idx`. Whatever the row's sparsity is, pruning always removes the **first `int(in_features * s_i)` columns** of this order (ties by smaller column index).

### 2. Iterating the per-row sparsities

Let \(Y=WX\) be the unpruned output and \(\hat Y(x)\) the output pruned by sparsity vector \(x\), starting from \(x_i=s\). Each round:

\[
q_i=\cos\bigl(Y_i,\hat Y_i\bigr),\quad
d_i=\frac{q_i-\min q}{\max q-\min q+10^{-6}},\quad
x\leftarrow x+2\eta\left(d-\overline d\right)
\]

then truncate and re-align:

\[
x\leftarrow \mathrm{clip}(x,0,0.95),\qquad
x\leftarrow x-\overline x+s
\]

The second step keeps the **average sparsity equal to \(s\)**: iteration only redistributes budget across rows. After each update the row is re-pruned, the output recomputed, and the flat cosine evaluated; only the \(x\) with the highest flat cosine is kept (`best_x`), together with its round (`best_iter`, 0-based).

### 3. Learning-rate grid

\(\eta\) is not adapted inside the loop but scanned outside:

- positive grid `[0.01, 0.02, 0.04, 0.08, 0.12, 0.16]`, in order; the scan **stops immediately** when a score does not beat the best score of this grid so far;
- the negative grid `[-0.01, -0.02, -0.04]` is tried only if the positive grid never beat the initial baseline and `forbid_neg_lr` is false;
- the best \(x\), `learning_rate` and `best_iter` are updated only on a **strictly greater** score. If no learning rate beats the baseline, `learning_rate` stays `None`.

### 4. Post-processing and mask

With the best \(x\), upstream additionally applies `>0.95→0.95`, `<0→0`, and again subtract-mean-add-\(s\), and only then builds the mask from `int(in_features * s_i)`. This runs **after** selection: the returned `best_metric`/`learning_rate` refer to the pre-post-processing \(x\).

## Task

Implement `solve(weight, inputs, activation_ms, target_sparsity, iterations=10, qmetric="cosim_flat", qmetric_dimwise="cosim", forbid_neg_lr=False, lr_grid=(0.01,...,0.16), neg_lr_grid=(-0.01,-0.02,-0.04))`:

- `weight`: `(out, in)` float32/float64, `out >= 2`;
- `inputs`: `(in, tokens)` same dtype, `tokens >= 1`;
- `activation_ms`: `(in,)` same dtype, the per-column mean square used for ordering;
- `target_sparsity`: in `(0, 1)`; `iterations >= 0`;
- only `qmetric="cosim_flat"` and `qmetric_dimwise="cosim"` are supported; anything else raises `ValueError`.

All arithmetic is float64. Cosines use a `1e-8` norm guard and are clamped to `[-1, 1]`; `flat` means the whole layer, `dimwise` means per output row; `starting_var`/`ending_var` are the **unbiased** (`ddof=1`) variance of the per-row cosines under the initial/best sparsities.

Return a 9-key dict:

- `sparsities`: `(out,)` float64, the **post-processed** per-row sparsities;
- `keep_mask`: `(out, in)` bool, True means kept;
- `pruned_weight`: `(out, in)` in the input dtype;
- `baseline_metric`, `best_metric`, `starting_var`, `ending_var`: python floats;
- `learning_rate`: the selected rate, possibly `None`;
- `best_iter`: python int.

## Example

```
weight = [[3,2,1,0],[0,1,2,3],[1,1,1,1],[4,0,0,1]]
inputs = [[1,0,1],[0.5,1,-1],[2,0.5,1],[1,1.5,-1]]   # (in=4, tokens=3)
activation_ms = [2/3, 0.75, 1.75, 17/12]
target_sparsity = 0.5, iterations = 5
```

After ordering, 5 rounds run over the positive learning-rate grid; `0.01` beats the baseline immediately:

```
baseline_metric = 0.9856996457     best_metric = 0.9928880902
learning_rate   = 0.01             best_iter   = 0
starting_var    = 8.6939e-06       ending_var  = 1.2014e-18
sparsities      = [0.48964132, 0.49108361, 0.50963753, 0.50963754]
keep_mask       = [[T,T,T,F],
                   [F,T,T,T],
                   [F,F,T,T],
                   [T,F,F,T]]
pruned_weight   = [[3,2,1,0],
                   [0,1,2,3],
                   [0,0,1,1],
                   [4,0,0,1]]
```

The four sparsities average exactly `0.5`: iteration only moves budget from insensitive rows to sensitive ones. Row 2's `0.5096` is harsher than row 0's `0.4896`, meaning it was judged easier to prune in this round.

## Constraints

- **Order once**: only sparsities change per round; the metric is not re-sorted, so the pruned set is always a prefix of each row's `sorted_idx`.
- **Budget conservation**: both the in-loop and post-processing steps subtract the mean and add the target, so the final mean equals `target_sparsity` (provided the 0.95/0 clamps do not alter it; they are followed by another re-centering).
- **Strict improvement**: equal scores do not update the best; `learning_rate` may be `None`.
- **Stopping rule**: as soon as one \(\eta\) does not beat the current best of its grid, the scan stops; the negative grid runs only when the positive grid achieved nothing.
- **Variance convention**: `torch.var` is unbiased by default (`ddof=1`), and `out=1` yields NaN, hence `out >= 2`.
- **Post-processing after selection**: do not back-propagate the post-processed sparsities into `best_metric`/`learning_rate`.
