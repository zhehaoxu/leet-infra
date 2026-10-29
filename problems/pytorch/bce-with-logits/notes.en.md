# Explained: BCE with Logits (pos_weight and Reduction Modes)

## Origin

The reason `F.binary_cross_entropy_with_logits` exists is stated plainly in the PyTorch docs: "numerically more stable than calling `binary_cross_entropy` after a sigmoid". Multi-label classification (COCO annotations, open-vocabulary detection), click-through models and binary mask losses all use it; `pos_weight` is the first lever when positives are extremely rare (positive rates of order $10^{-3}$), and the "positive class weight" of a detection head is exactly this argument.

## From the Probability Form to the Stable Form

With $p=\sigma(x)$, the elementwise loss (including `pos_weight` $w$) is

$$\ell = w\,y\,-\log p + (1-y)\,-\log(1-p) = w\,y\log(1+e^{-x}) + (1-y)\log(1+e^{x})$$

The key elementary identity (verified directly on the branches $x>0$ and $x\le 0$):

$$\log(1+e^{t}) = t^{+} + \underbrace{\log(1+e^{-|t|})}_{L}, \qquad t^{+}=\max(t,0)$$

The exponent argument inside $L$ is always $-|t|\le 0$, so $e^{-|t|}\in(0,1]$ and `log1p` is accurate; for large $|t|$ it underflows to 0, and 0 is precisely the correct asymptote of $\log(1+e^{-|t|})$. Substituting and using $\max(-x,0)=\max(x,0)-x$:

$$\ell = w\,y\big(x^{+}-x\big) + (1-y)x^{+} + \big(w y + 1 - y\big)L = (1-y)x^{+} + w\,y(x^{+}-x) + \big(1+y(w-1)\big)L$$

With $w=1$ this collapses to the textbook $\max(x,0)-xy+L$. The reference solution is that last line turned literally into tensor ops, with no `sigmoid` anywhere.

## The Denominators of the Three Reduction Modes

| reduction | denominator | note |
| --- | --- | --- |
| `none` | — | elementwise, already multiplied by `weight` |
| `sum` | 1 | |
| `mean` | `logits.numel()` | **not** $\sum\text{weight}$ |

The `mean` row is a genuine engineering dispute: dividing by $\sum w$ (a weighted average) looks more "mean-like" statistically, but PyTorch divides by the element count so that `mean` and `sum` differ only by a constant factor and the gradient scale stays predictable. The judge verifies this with an all-3 `weight`: `mean(w=3) == 3·mean(w=None)`, whereas dividing by $\sum w$ would cancel it back to 1× — anyone who implements the weighted average fails here. An all-ones `weight` must match `weight=None` pointwise.

## Core Points and Pitfalls

1. **`pos_weight` scales only the positive term**: the $y=0$ term carries no $w$. The equivalent check `-(w*y*logsigmoid(x) + (1-y)*logsigmoid(-x)).sum()` appears in the judge.
2. **`pos_weight` is a broadcast tensor**: both the scalar tensor `[2.0]` and a per-class vector `rand(5)` must work, via elementwise broadcasting — never `.item()`.
3. **`weight` multiplies the loss**: `(loss * weight)`, not the logits and not the targets.
4. **`reduction` matches strings exactly**: `"MEAN"` is case sensitive; PyTorch raises ValueError and the judge requires the same.
5. **Keep intermediates in float64**: at `x=±1000`, adding $x^{+}$ and $L$ in float32 drops $L$ (adding $10^3$ and $10^{-434}$); there $L$ is 0 anyway, but for the $x=\pm100$ segment $L\sim 3.7\times10^{-44}$ only survives in float64, and the atol=1e-6 tolerance is granted on the assumption of float64 accumulation.

## How It Differs from Neighbouring Problems

- `bce-loss`: takes **probabilities** $p\in[0,1]$, hence needs clamping and is inherently unstable; this problem takes logits and must never apply a sigmoid.
- `logsumexp` (order360): the same family of stabilisation trick (factoring out a maximum), but there it is a normaliser and here a softplus.
- `focal-loss` / `dice-loss`: both reweight or measure overlap on probabilities; this is a per-sample binary log loss.
- `cross-entropy-loss`: the multi-class $\log\sum e$ convention; equivalent to this problem in the two-class case but with different reduction semantics.

## Cross-Checking and Complexity

- The oracle is `F.binary_cross_entropy_with_logits` over the full Cartesian product of `weight` (None / same-shape random), `pos_weight` (None / scalar tensor / per-class vector) and the three `reduction` modes; plus large-logit finiteness probes, the `mean` denominator probe, the analytic value $\log 2$, and four illegal `reduction` values.
- Complexity $O(N)$: one `exp` and one `log1p`, no overhead beyond the reduction itself.
