# logsumexp: Max-Subtraction and Degenerate ±inf Segments

## Problem Description

$\log\sum e^{x}$ is the normaliser hiding inside softmax, cross-entropy, variational lower bounds and sequence likelihoods. The naive `log(sum(exp(x)))` breaks in two places: with $x\sim 10^3$ the `exp` overflows to `inf`, and with $x\sim -10^3$ everything underflows to 0 so the `log` gives $-\infty$. The standard fix factors out the maximum, pushing the exponent argument into $(-\infty, 0]$:

$$\log\sum_i e^{x_i} = m + \log\sum_i e^{\,x_i - m}, \qquad m = \max_i x_i$$

The second hurdle is degenerate segments: when a mask sets a whole row of logits to $-\infty$ then $m=-\infty$ and $x_i - m$ becomes $-\infty-(-\infty)=\mathrm{NaN}$, poisoning the segment; and a single $+\infty$ inside a segment must make the result $+\infty$. Both branches need explicit values — the stable form does not survive them on its own.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- `dim=None` is an extension beyond torch (`torch.logsumexp` requires a dim): reduce everything into a **0-dimensional** tensor, and with `keepdim=True` keep **every axis as length 1**.
- For an int or tuple-of-ints `dim` the behaviour matches `torch.logsumexp` exactly, negatives folded against `x.dim()`; out of range, duplicate axes or an empty tuple raise ValueError.
- Degenerate conventions: an all $-\infty$ segment → $-\infty$ (**not NaN**); a segment containing $+\infty$ → $+\infty$; a segment containing NaN → NaN propagates.
- `x.exp().sum().log()` is forbidden; compute internally in float64 and return float32.

## Example

$X$ is $2\times 3$ (numbers taken from the EXAMPLE_* constants filled in from test_cases.py):

```
X = [[   0,    1,    2],
     [-100,  -99,  -98]]

dim=1 -> [  2.4076058864593506, -97.59239196777344]
dim=0 -> [  0.0,  1.0,  2.0]
(0,1) ->  2.4076058864593506
```

Row 0 has $m=2$, so $2 + \log(e^{-2}+e^{-1}+1) = 2 + \log(1.50827) = 2.4076$; row 1 has $m=-98$ and the shift keeps the structure "$m$ plus a correction in $[0,\log 3]$", so the answer is $-97.5924$ rather than $-\infty$.

## Constraints

- Input is float32 and may contain $-\infty$ / $+\infty$ / NaN; compared against `torch.logsumexp` with atol=1e-5, rtol=1e-5
- The judge probes the $10^3$ and $-10^3$ segments (asserting first that the naive form really fails there), cross-validates `log_softmax(x) = x − logsumexp(x)`, and checks illegal `dim` values
