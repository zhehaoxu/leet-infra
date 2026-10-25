# Causal Mask Explained

## Definition

In an autoregressive model (GPT-style), position $i$ may only see tokens with index $\le i$ and must not "peek at the future". This is implemented by setting the **strict upper triangle** of the attention score matrix to $-\infty$:

$$ \text{mask}_{ij}=\begin{cases}0,& j\le i\\ -\infty,& j>i\end{cases},\qquad \text{scores}+\text{mask} $$

After softmax, $-\infty$ becomes a weight of 0, which is equivalent to forbidding attention to future positions.

## Keep vs mask direction (continuing the SDPA semantics)

- `keep = torch.tril(ones(bool))`: **lower triangle True = allowed**, with the diagonal included (position $i$ can see itself).
- Apply with `masked_fill(~keep, -inf)`: fill the upper triangle (the future) with $-\infty$. Reversing the direction turns it into "may only see the future".

## Why use -inf rather than a large negative number

softmax is shift-invariant, so in theory $-10^9$ also approximately masks, but $-\infty$ gives exactly 0 weight mathematically. The risk: if some row is **entirely** $-\infty$ (which does not occur in a standard lower-triangular causal mask, since the first row at least keeps the diagonal), it produces NaN. A standard causal mask keeps at least one position per row, so it is safe.

## Combining with other masks

The padding mask (masking pad tokens, see the SDPA problem) and the causal mask are often used **together**: `keep = causal & pad`. During decoder training the two are stacked.

## Size and device

The mask has shape $(L,L)$, and its `device` must follow `scores`, otherwise cross-device operations error. Remember to pass `device=scores.device` to `torch.full`/`ones`.
