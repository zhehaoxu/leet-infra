# Sparsification: OWL Outlier-Ratio Layer-Wise Sparsity Allocation

## Problem

In one-shot pruning a **uniform sparsity across layers** is rarely optimal: some layers have many activation outliers, so their weight×activation scores are dominated by a few large entries and trimming the tail is cheap; other layers have smooth activations and should be pruned less. OWL (*Outlier Weighed Layerwise Sparsity*, 2310.05175) scores each layer with an observable statistic and then redistributes the global budget accordingly.

### 1. Per-layer metric and outlier ratio

For every Linear in a layer, compute the same score as Wanda, \(m_{ij}=|W_{ij}|\cdot\sqrt{\mathbb{E}_t[X_{t,j}^2]}\). **Flatten and concatenate the metrics of all Linears in the layer**, let \(\mu\) be the overall mean, and define the outlier ratio

\[
r_l = \frac{\#\{\,m > \mu\cdot \texttt{hyper\_m}\,\}}{\#\{m\}}\times 100
\]

The comparison is **strictly greater than**, and the threshold is "mean × `hyper_m`" rather than "mean + k standard deviations" (upstream also has `check_outlier` using the max, but this path does not call it).

### 2. Normalisation and budget alignment

Min-max normalise the \(L\) ratios onto \([0, 2\lambda]\):

\[
\tilde r_l=\frac{r_l-r_{\min}}{r_{\max}-r_{\min}}\cdot 2\lambda
\]

then subtract the mean and add \(1-s\) (with \(s\) the global target sparsity):

\[
d_l=\tilde r_l-\overline{\tilde r}+(1-s),\qquad \text{layer\_sparsity}_l=1-d_l
\]

Hence \(\sum_l \text{layer\_sparsity}_l/L = s\): normalisation only redistributes the budget. If a layer comes out with `layer_sparsity <= 0` (density above 1), upstream replaces it with `0.01` — a hard 1%-sparsity floor, not a `1-...` expression.

### 3. Within-layer mask

With `layer_sparsity_l` in hand, each Linear is pruned exactly like Wanda: stable-sort each row's metric ascending and prune the first `int(in_features * layer_sparsity_l)`, breaking ties by smaller column index. All Linears in one layer share the layer ratio.

## Task

Implement `solve(weights, activation_ms, sparsity, lamda=0.08, hyper_m=3.0)`:

- `weights`: nested `[layer][Linear]` list of `(out, in)` float32/float64 tensors;
- `activation_ms`: same structure, each entry an `(in,)` tensor with the input-channel **mean square** \(\mathbb{E}_t[X_{t,j}^2]\);
- `sparsity`: numeric in `[0,1]`; `lamda > 0`; `hyper_m > 0`;
- at least two layers, every layer non-empty, all tensors of one dtype.

Compute the ratios, densities and layer sparsities above, then build a mask per Linear. Return a 6-key dict:

- `outlier_ratio`, `density`, `layer_sparsity`: `(L,)` float64;
- `metric`, `mask`, `pruned_weight`: nested lists matching the input, with `metric`/`pruned_weight` in the input dtype and `mask` bool, True meaning pruned.

**Raise `ValueError`** for: fewer than two layers, empty layers, structure/dtype mismatches, activation length not equal to `in_features`, out-of-range `sparsity`, non-positive `lamda`/`hyper_m`, and **identical outlier ratios across layers** — upstream divides by `max-min = 0` there, so this problem declares it invalid.

## Example

Three layers, one `1×8` Linear each, all activations 1, `sparsity=0.5`, `lamda=0.08`, `hyper_m=3`:

```
layer 0 weights [1,1,1,1,1,1,1,1]    mean 1     threshold 3      0 outliers → r = 0
layer 1 weights [1,1,1,1,1,1,1,40]   mean 5.875 threshold 17.625 1 outlier  → r = 12.5
layer 2 weights [1,1,1,1,1,1,40,40]  mean 10.75 threshold 32.25  2 outliers → r = 25
```

```
r        = [0, 12.5, 25]
normalised = [0, 0.08, 0.16]          # (r-min)/(max-min) * 2*0.08
minus mean = [-0.08, 0, 0.08]         # mean 0.08
density  = [0.42, 0.5, 0.58]          # + (1-0.5)
sparsity = [0.58, 0.5, 0.42]          # 1-density
k        = [4, 4, 3]                  # int(8 * sparsity)
```

Layer 2 has the most outliers and receives the lowest sparsity 0.42 (only 3 columns pruned); layer 0 has none and is pruned hardest (4 columns). The masks are `[T,T,T,T,F,F,F,F]`, `[T,T,T,T,F,F,F,F]` and `[T,T,T,F,F,F,F,F]`.

## Constraints

- **Concatenate metrics per layer**: multiple Linears in a layer are flattened together before computing the mean and threshold.
- **Strictly greater**: an entry exactly equal to `mean*hyper_m` is not an outlier; the tests pin this with `[1,1,1,9]`.
- **Normalisation only redistributes**: the mean of `layer_sparsity` equals the target `sparsity` unless the 0.01 floor kicks in.
- **The floor is `0.01`**: when `layer_sparsity <= 0`, it becomes `0.01`; `int(in*0.01)` may then be 0, pruning nothing in that layer.
- **Ties go to the smaller index**: within a layer the stable ascending sort is used, as in the Wanda problem.
