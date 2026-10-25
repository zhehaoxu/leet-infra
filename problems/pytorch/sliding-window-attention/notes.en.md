# Sliding-Window Attention Explained

## Mask = causal AND window

The $j$ visible to position $i$ must satisfy two conditions at once:

$$ j \le i \quad(\text{causal}),\qquad i - j < \text{window}\quad(\text{lookback cap}) $$

Combining them yields a lower-triangular diagonal band of width $w$. Broadcasting a 1-D `idx` into an $(L,L)$ comparison matrix is the simplest approach:

```
causal = idx[:, None] >= idx[None, :]
within = (idx[:, None] - idx[None, :]) < window
mask   = causal & within          # True = visible
```

Note this collection adopts the convention "bool mask **True = visible**" (consistent with SDPA);
`masked_fill(~mask, -inf)` is what actually masks out — reversing it would treat the upper triangle as visible.

## Common Pitfalls

- **Use `<` rather than `<=` at the window boundary**: `i-j < window` makes the window contain exactly `window` tokens (including itself);
  writing `<=` looks back one token too many.
- **The diagonal is always visible**: `window>=1` guarantees $i-j=0<window$, so every row has at least one non-$-\infty$ entry, and
  softmax never degenerates. Allowing `window=0` would produce all-$-\infty$ rows → NaN.
- **Degeneration relation**: when `window >= L`, `within` is always true, equivalent to pure causal attention — usable as a regression case.
- **Locality probe**: perturbing the v of a far-away token should affect only the `window` positions after it; this catches "did causal but
  forgot the window" half-correct implementations.

## Applications & Extensions

Mistral 7B uses a sliding window with `window=4096`; tokens beyond the window rely on **inter-layer stacking** to widen the receptive field
(each layer sees 4096, and after several layers the effective context far exceeds the window); Longformer instead mixes "sliding window + a few global tokens".
The sliding window lets the KV cache keep only the most recent `window` tokens, which is key to saving memory in long-sequence inference.
