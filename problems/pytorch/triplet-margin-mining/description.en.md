# Triplet Margin Loss with Online Hard Example Mining (batch_hard / semihard)

## Problem

A batch of \(N\) samples yields \(O(N^3)\) triplets — using all of them is too expensive and sampling them at random is too weak. FaceNet (Schroff et al., CVPR 2015, arXiv:1503.03832) proposed **semihard** mining and batch-hard mining (Hermans et al., *In Defense of the Triplet Loss for Person Re-Identification*, arXiv:1703.07737) proposed **batch_hard**; both are "online" schemes: instead of being handed triplets, they get only (embeddings, labels) and let each anchor pick its own positive and negative inside the batch. This problem implements both mining strategies plus the triplet margin loss, including PyTorch's `swap` variant, returning a single 0-dim scalar. CPU, fully deterministic, no random number calls; distance ties always take the **smallest index**, so judging is exactly reproducible.

## Rules

- The complete signature, argument meanings and defaults live only in the starter docstring; that is the sole judging contract. embeddings is (N,D) float (N=0 allowed), labels is (N,) long; strategy ∈ {`"batch_hard"`, `"semihard"`}, anything else raises ValueError.
- **Pairwise distance**: \(D_{ij}=\lVert x_i-x_j\rVert_p\), computed as "elementwise difference, take \(|\cdot|^p\), sum, then raise to \(1/p\)" (\(p=2\) Euclidean, \(p=1\) Manhattan; not the approximation path of a distance-matrix operator).
- **Mining (for each anchor \(a\))**:
  - Positive pool \(\{j:\text{labels}[j]=\text{labels}[a],\ j\ne a\}\): take the **largest** distance as pos (ties → smallest \(j\)); empty pool → **skip the anchor**.
  - Negative pool \(\{j:\text{labels}[j]\ne\text{labels}[a]\}\) (the anchor's own label matches itself, so it is excluded automatically):
    * `batch_hard`: take the **smallest** distance as neg (ties → smallest \(j\));
    * `semihard`: prefer the smallest distance inside the **open interval** \((d_{ap},\,d_{ap}+\text{margin})\) (ties → smallest \(j\)); if that interval is empty, **fall back** to the smallest distance over the whole negative pool (i.e. the batch_hard pick).
  - Empty negative pool → skip the anchor.
- **Per-triplet loss** (the finalized formula, checked against `F.triplet_margin_loss` on fixed triplets):
  \[\text{swap=False: }\ \ell_a=\operatorname{relu}(\text{margin}+d(a,\text{pos})-d(a,\text{neg})),\qquad
    \text{swap=True: }\ \ell_a=\operatorname{relu}(\text{margin}+d(a,\text{pos})-\min\{d(a,\text{neg}),\,d(\text{pos},\text{neg})\}).\]
  **The min acts on the negative side**, it does not shrink \(d_{ap}\).
- **Return**: the mean loss over all **valid anchors** (those not skipped), as a 0-dim scalar tensor; 0 when there is no valid anchor. Skipped anchors are **not in the denominator**.

## Example

embeddings \(=[[0,0],[3,0],[0,2],[1,2]]\), labels \(=[0,0,1,1]\), margin=1.0, p=2.0, swap=False. Distance matrix (partial): \(d_{01}=3\), \(d_{02}=2\), \(d_{03}=\sqrt5=2.2361\), \(d_{12}=\sqrt{13}=3.6056\), \(d_{13}=\sqrt8=2.8284\), \(d_{23}=1\).

| anchor | pos (farthest same-class) | batch_hard neg (nearest other-class) | \(\ell_a\) | semihard neg (nearest inside the interval, else fallback) | \(\ell_a\) |
|---|---|---|---|---|---|
| 0 | 1, \(d_{ap}=3\) | 2, \(d_{an}=2\) | \(\operatorname{relu}(1{+}3{-}2)=2\) | interval \((3,4)\) empty → fallback 2, \(d_{an}=2\) | 2 |
| 1 | 0, \(d_{ap}=3\) | 3, \(d_{an}=\sqrt8=2.8284\) | \(1.1715729\) | interval \((3,4)\) contains \(d_{12}=\sqrt{13}\) → pick 2 | \(0.3944487\) |
| 2 | 3, \(d_{ap}=1\) | 0, \(d_{an}=2\) | \(\operatorname{relu}(0)=0\) | interval \((1,2)\) empty → fallback 2 | 0 |
| 3 | 2, \(d_{ap}=1\) | 0, \(d_{an}=\sqrt5\) | 0 | interval \((1,2)\) empty → fallback 0, \(d_{an}=\sqrt5\) | 0 |

Means (backfilled EXAMPLE_*):

- `batch_hard` = \((2+1.1715729+0+0)/4=\) **0.7928932309**
- `semihard` = \((2+0.3944487+0+0)/4=\) **0.5986121893**

semihard diverges from batch_hard only at anchor 1: it picks the farther \(\sqrt{13}\) instead of the nearest \(\sqrt8\), so the loss shrinks — exactly the design intent of "semi-hard" (avoid pathologically close negatives that may be mislabels).

## Constraints

- strategy accepts only `"batch_hard"`/`"semihard"`, anything else ValueError; margin and p are plain reals (\(p=1.0\) and \(p=2.0\) are the main conventions, passed as floats)
- Distance ties (including ties inside the semihard interval) **always take the smallest index**; this is what makes the judging reproducible
- An anchor whose positive or negative pool is empty is skipped: all-same-label, all-distinct-label, N≤1 and N=0 all return scalar 0
- The return must be a 0-dim scalar tensor (not a Python float, not a (1,) tensor); float32 tolerance \(10^{-5}\)
- Agreement with `F.triplet_margin_loss` is defined for "same triplets, same p/margin/swap"; this problem adds three conventions that `F` does not define: mining, skipping, and the interval-empty fallback
