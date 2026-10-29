# BCE with Logits: pos_weight and Reduction Modes

## Problem Description

Multi-label classification, OCR/detection confidence heads and click models all use binary cross entropy, and in practice one must **never** apply a sigmoid first and then take the log: once logits reach $\pm10^3$ the `sigmoid` saturates to 0 or 1, `log 0` yields $-\infty$ and training turns NaN at the first large logit. PyTorch's `binary_cross_entropy_with_logits` fuses the two into an algebraically identical but numerically stable form:

$$\ell_i = -\Big[y_i\log\sigma(x_i) + (1-y_i)\log\big(1-\sigma(x_i)\big)\Big]$$

Using $\log\sigma(x) = -\log(1+e^{-x})$, $\log(1-\sigma(x)) = -\log(1+e^{x})$ and the identity $\log(1+e^{t}) = t^{+} + \log(1+e^{-|t|})$ one arrives at an expression that cannot overflow (with $L=\log(1+e^{-|x|})$ and $w$ the `pos_weight`, defaulting to 1):

$$\ell_i = (1-y_i)\max(x_i,0) + w_i\,y_i\big(\max(x_i,0) - x_i\big) + \big(1 + y_i(w_i - 1)\big)L$$

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- Sigmoid-then-log is forbidden; the judge probes $x\in\{\pm10^3,\pm10^2,0\}$ and asserts every output is finite.
- `reduction`: `"none"` returns the elementwise loss (**already multiplied by `weight`**, shaped like logits); `"sum"` sums everything; `"mean"` is **sum ÷ logits.numel()** — the denominator is not $\sum\text{weight}$.
- `pos_weight` only scales the positive-class term (the $y=1$ term is multiplied by $w$) and may broadcast against logits; `weight` is an elementwise sample weight applied to the whole loss.
- Any other `reduction` value (including `"MEAN"`, `None`, the empty string) raises ValueError.

## Example

$x=[-2,-0.5,0,1,3]$, $y=[1,0,1,0,1]$ (numbers taken from the EXAMPLE_* constants filled in from test_cases.py):

```
loss(none) = [2.1269280910491943, 0.4740769863128662, 0.6931471824645996,
              1.31326162815094,   0.04858735203742981]
mean       = 0.9312002658843994
sum, pos_weight=2 -> 7.524663925170898
```

The first entry has $x=-2, y=1$: $-\log\sigma(-2)=\log(1+e^{2})=2.1269$; the third has $x=0,y=1$: $-\log 0.5=\log 2=0.6931$. The `mean` denominator is the 5 elements; `pos_weight=2` doubles the three $y=1$ terms and sums to $7.5247$.

## Constraints

- Input is float32, computed in float64, returned as float32; compared against `F.binary_cross_entropy_with_logits` with atol=1e-6, rtol=1e-5
- `weight` / `pos_weight` appear as None, a scalar tensor and a per-class vector; an all-ones `weight` must match `weight=None` pointwise
- Division of labour with `bce-loss` (probability input): that problem takes values already passed through a sigmoid, while here a sigmoid must never appear.
