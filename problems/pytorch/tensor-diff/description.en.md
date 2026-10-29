# diff: n-th Order Difference with prepend/append

## Problem Description

The adjacent difference is the shortest snippet in time-series feature engineering, gradient-consistency checks, discretisation and "convexity from the second difference": `x[1:] - x[:-1]`. `torch.diff` turns it into an operator with an order, boundary concatenations and any axis — while hand-written versions usually only work for `dim=-1`, or miscount the output length by one.

$$\Delta^{(1)} x_i = x_{i+1} - x_i, \qquad \Delta^{(n)} x = \Delta^{(1)}\big(\Delta^{(n-1)} x\big)$$

Along `dim`, `prepend` is attached at the head and `append` at the tail, giving a sequence of length $L = \dim_x + \dim_p + \dim_a$; after $n$ differencing steps the output length is $L - n$.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- `n` is a non-negative int, default 1; `n=0` returns a **new tensor with identical content** (not the same object); a negative or non-int `n` raises ValueError.
- `dim` defaults to `-1`, negatives folded against `x.dim()`; out of range raises ValueError. Middle axes (`dim=0`, `dim=1`) must be exactly as correct as the last one.
- `prepend` / `append` must have the same ndim as `x`, agree on every dimension except `dim`, and share its dtype, otherwise ValueError.
- If the length along `dim` after differencing is $\le 0$, raise ValueError (torch returns an empty tensor there; this problem requires non-empty so the judging signal stays meaningful).

## Example

$X$ is $2\times 4$ (numbers taken from the EXAMPLE_* constants filled in from test_cases.py):

```
X       = [[ 1,  4,  9, 16],
           [ 2,  3,  5,  9]]

n=1     = [[ 3,  5,  7],
           [ 1,  2,  4]]
n=2     = [[ 2,  2],
           [ 1,  2]]
dim=0   = [  1, -1, -4, -7]
prepend0= [[ 1,  3,  5,  7],     append0= [[ 3,  5,  7, -16],
           [ 2,  1,  2,  4]]              [ 1,  2,  4,  -9]]
```

`n=2` is "the difference of the difference": $5-3=2$, $7-5=2$. Prepending 0 makes the first entry $1-0=1$ with length still $5-1=4$; appending 0 makes the last entry $0-16=-16$. For the squares the first difference is the odd sequence $2i+1$ and the second difference is the constant 2 — precisely these two rows.

## Constraints

- Input is float32 and the output dtype matches it; compared pointwise against `torch.diff` with atol=1e-6
- Coverage: $n\in\{0,1,2,3,4,6\}$, positive and negative `dim` spellings, `prepend`/`append` on one side or both at once
- Illegal `n` (negative, float), out-of-range `dim`, mismatched `prepend`/`append` shape or dtype, and a non-positive resulting length must all raise ValueError
