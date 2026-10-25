# Temperature Scaling & Top-p Sampling (Temperature + Nucleus Sampling)

## Problem

At decoding time, logits first go through a **temperature-scaled** softmax that controls how sharp the distribution is, and then through **top-p (nucleus) truncation**: only the head set whose descending cumulative probability first reaches $p$ is kept, everything else is set to 0 and renormalized.

$$ \text{softmax}_T(z)_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}} $$

- $T \to 0$: the distribution approaches the one-hot argmax (more deterministic);
- $T \to \infty$: the distribution approaches uniform (more random).

## Rules

- Implement `solve(logits, temperature=1.0, top_p=None)`, returning probabilities row by row along the last dimension, with each row summing to 1.
- The temperature acts on the logits **before** the softmax: $z/T$. You must subtract the per-row maximum for numerical stability (large logits must not overflow).
- **top-p nucleus** (when `top_p` is given and $<1$): sort each row by descending probability, take the `cumsum` of the probabilities, and drop the tail entries satisfying `(cumsum - own_probability) >= top_p` — that is, **keep the head set whose cumulative probability first reaches $p$**, and in any case **keep at least 1 entry** (when the largest single probability is already $\ge p$, only 1 entry remains). The kept entries are renormalized, the dropped entries become 0, and the output is restored to the original order.
- When `top_p=None` or `>=1.0`, no truncation is applied.

## Examples

```
Input: logits=[[2,1,0.5,0.1]], temperature=1.0
Output: [[0.5745217, 0.2113547, 0.1281931, 0.0859304]]   # plain softmax

Input: logits=[[10,9,0,-5]], temperature=1.0, top_p=0.9
   The first two probabilities of the softmax sum to ≈1.0, so the cumulative already exceeds 0.9 from the third entry on → truncate the tail
Output: [[0.7310585, 0.2689414, 0.0, 0.0]]   # keep two entries and renormalize
```

## Constraints

- float32/float64 input, arbitrary leading batch dimensions supported
- Even for an extremely small `top_p`, at least 1 entry must be kept (a row of all zeros is not allowed)
