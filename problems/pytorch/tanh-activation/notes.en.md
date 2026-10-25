# Tanh Explained

## Definition

$$ \tanh(x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}=2\sigma(2x)-1 $$

It compresses the reals into $(-1,1)$ and is **zero-centred, an odd function**.

## Relation to sigmoid

Same family, differing by one affine map: $\tanh(x)=2\sigma(2x)-1$. So tanh's output mean is closer to 0 and its gradient is stronger than sigmoid's ($2\times$ slope), which is why early RNNs preferred tanh.

## Implementation and numerical details

- `torch.tanh` is already numerically stable: as $x\to\pm\infty$ it saturates to $\pm1$ without overflowing.
- If you hand-write $(e^{2x}-1)/(e^{2x}+1)$, then for large $x$ the $e^{2x}$ overflows → `inf/inf=NaN`. Split positive/negative branches before calling `torch.exp`, or just use the library function.

## Gradient

$$ \tanh'(x)=1-\tanh^2(x) $$

The gradient is 1 (its maximum) at $x=0$ and tends to 0 at both ends — like sigmoid it is a saturating activation, so deep stacks or saturated inputs cause vanishing gradients. LSTM/GRU use sigmoid for the gates and tanh for the state candidate, which is exactly taking the strengths of each.
