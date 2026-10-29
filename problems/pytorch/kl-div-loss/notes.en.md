# Explained: kl_div Loss (Log-Prob Input and the batchmean Convention)

## Origin

The shape of `F.kl_div`'s interface comes straight from the distillation pipeline: in Hinton et al. 2015 (1506.04420) the student side computes `log_softmax` while the teacher supplies probabilities; were the function to take $q$ instead of $\log q$ it would internally log an extremely small $q$ again (below $10^{-45}$ a float32 probability is already 0). Accepting $\log q$ as input sidesteps that, which is why both PyTorch and Keras settled on the awkward-but-sound "first argument is a log-probability" convention. `batchmean` was added later precisely so that "per-sample KL averaged over the batch" matches the mathematics; the default remains `"mean"` for backwards compatibility, and that mismatch is this problem's subject. On the RLHF side, `to-kl-controller` / the PPO KL penalty and the VAE β-regulariser all consume this convention.

## Signs and Degenerate Cases of the Elementwise Term

$$\ell_c = p_c(\log p_c - \log q_c)$$

Each boundary needs a definite value:

| condition | $\ell_c$ | implementation note |
| --- | --- | --- |
| $p_c > 0,\; q_c > 0$ | $p_c(\log p_c - \log q_c)$, either sign | ordinary case |
| $p_c > 0,\; q_c = 0$ | $+\infty$ | $\log q = -\infty$, the subtraction yields $+\infty$ naturally |
| $p_c = 0$ | $0$ (convention $0\log 0 := 0$) | needs an explicit mask, otherwise $0\cdot(-\infty) = \mathrm{NaN}$ |

The reference handles the third row with a single `where`: `torch.where(t > 0, t * (t.log() - q), zeros)`. With `log_target=True`, $p=0$ appears as `target = -inf`, so the predicate becomes `isinf(t) & (t < 0)`. The statement explicitly **does not copy** the `F.kl_div` detail that returns NaN when $p=0$ and $q=0$ — the judge asserts 0 for that cell.

## batchmean and Distillation Weights

With broadcast output shape $(B, C)$:

$$\text{batchmean} = \frac{1}{B}\sum_{b}\sum_{c}\ell_{b,c}, \qquad \text{mean} = \frac{1}{BC}\sum_{b,c}\ell_{b,c} = \frac{\text{batchmean}}{C}$$

Coefficients in formulas such as `loss = (1-α)·CE + α·β²·KL` are derived for the per-sample KL, so using `mean` shrinks the KL term by a factor $C$ (with a $150k$ vocabulary that amounts to silently switching the regulariser off). The judge pins this order-of-magnitude difference on a small case with $C=6$, asserting `batchmean == mean × 6`. Also, non-negativity of KL only holds **after** reduction (Gibbs' inequality) — elementwise terms can be negative, which is why the assertions use a same-batch pair and check `batchmean >= 0` instead of testing the `none` output for non-negativity.

## Core Points and Pitfalls

1. **Argument order**: `solve(log_pred, target)`; swapping them fails immediately on the dense random cases ($p$ and $\log q$ live at completely different magnitudes).
2. **`batchmean` divides by dim 0 of the output**: when `target` broadcasts to $(5,4)$ then $B=5$, not the length-1 of the passed `target`; the reference reads `elem.shape[0]`.
3. **`batchmean` requires at least one dimension**: a 0-d input has no "length of dim 0", so it raises ValueError.
4. **One-hot degeneration**: when $p$ is one-hot, $\sum_c \ell = -\log q_{c^*}$ is exactly cross entropy; the judge uses this identity as an independent check (linking to `cross-entropy-loss`).
5. **float64 matters**: $\log p$ for $p\sim10^{-8}$ is very sensitive to float32 precision, and the atol=1e-6 tolerance presumes float64 intermediates.

## How It Differs from Neighbouring Problems

- `kl-divergence`: given two probability vectors, compute the scalars $\mathrm{KL}(P\|Q)$ and $\mathrm{KL}(Q\|P)$ — the mathematical definition and its asymmetry.
- `jsd-loss`: the symmetrised Jensen–Shannon divergence, which needs KL against $M=(P+Q)/2$ on both sides.
- `cross-entropy-loss`: the special case of one-hot $p$; this problem's hard-label branch reproduces it exactly.
- `to-kl-controller` / `to-grpo-advantage`: the controller and coefficient side of using KL as an RLHF penalty, not the elementwise convention.

## Cross-Checking and Complexity

- The oracle is `F.kl_div` (pointwise on dense cases; the $p=0,q=0$ cell is asserted per the problem's own convention), plus the one-hot analytic check, the $q=0$ $+\infty$ case, a broadcast target, the 6× gap between `batchmean` and `mean`, a `log_target=True` round trip, and four illegal `reduction` values.
- Complexity $O(N)$ over the broadcast element count: two passes (the `where` and the `sum`).
