# Temperature-Scaled Softmax and Top-p Sampling Explained

## Temperature scaling

Divide the logits by the temperature $T$ before the softmax:

$$ p_i=\text{softmax}(z/T)_i $$

- $T>1$: a **flatter** distribution (more random, higher diversity);
- $T<1$: **sharper**, and as $T\to0$ it approaches the argmax one-hot; $T=1$ is the plain softmax.
- Still divide by the temperature first and **then subtract the row maximum** before the exp (numerical stability is unchanged, see the "Softmax" problem). The scaling does not care about the subtraction order, but `logits/T` must come before normalization.

## Top-p (nucleus) truncation

Sampling strategy: instead of a fixed top-k, take the smallest candidate set (the nucleus) by a **cumulative probability threshold** $p$, discard the long tail, then renormalize inside the nucleus. Compared with top-k, the number of candidates adapts to the sharpness of the distribution.

Implementation (this reference solution) key points:

1. First compute `probs` via temperature softmax, **sort in descending order** and record the indices `idx`.
2. Accumulate `cum = cumsum(sorted)`; decide "keep" using
   $$ \text{remove}=(\text{cum}-\text{sorted})\ge p $$
   i.e. entries whose cumulative **excluding themselves** is still $<p$ stay — this **guarantees at least 1 entry is kept** (otherwise, if the first entry's own value is already $\ge p$, everything would be deleted). This is the easiest line to get wrong.
3. Set the removed positions to 0, renormalize the kept ones, and use `scatter_(-1, idx, kept)` to restore the original order.

## Contract reminders

When `top_p>=1` or `None`, do not truncate and return the temperature distribution directly. `cum - sorted` is "the cumulative before excluding self"; writing `cum >= p` would also drop the entry that exactly crosses the threshold, disagreeing with the HuggingFace implementation.
