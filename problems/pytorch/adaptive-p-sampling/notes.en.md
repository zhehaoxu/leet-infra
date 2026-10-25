# Solution Notes · Adaptive-p Sampling

## Reshaping rather than truncation

adaptive-p encodes "sampling preference" into a **peak function**: centered at the target probability $p^\star$, the closer a token's original probability
is to $p^\star$, the higher its new logit.

$$\text{new\_logit}_i = \text{PEAK} - \text{SHARPNESS}\cdot\frac{\text{dist}_i^2}{1+\text{dist}_i},\quad
\text{dist}_i = |p_i - p^\star|\cdot \text{INV\_WIDTH}$$

- Near the peak, the $\text{dist}^2$ term gives **quadratic** fine-grained separation (pulling apart the few tokens closest to $p^\star$).
- Far away, the $/(1+\text{dist})$ makes the growth degrade to **linear**, preventing the far end from blowing up under softmax while not zeroing it out completely.
- No candidate is ever removed; `filtered` covers the whole vocabulary — this is the essential difference from top-p/min-p.

## Feedback through adapted_target

The EMA records the weighted average of the **original** probability of the token actually drawn at each past step,
$\bar p = \text{weighted\_sum}/\text{total\_weight}$. The target is written as

$$p^\star = \text{clamp}(2\cdot\text{target} - \bar p,\ 0, 1)$$

Thus a high $\bar p$ (historically too sharp a distribution) → $p^\star$ is pushed down → the peak moves into the low-probability region → the next distribution is flatter,
and vice versa. This forms a negative feedback loop around `target`. When `total_weight==0` (the first step), $p^\star=\text{target}$.

## EMA update (the most error-prone part)

$$\text{weighted\_sum}' = p_{\text{orig}}[\text{chosen}] + \text{decay}\cdot\text{weighted\_sum},\quad
\text{total\_weight}' = 1 + \text{decay}\cdot\text{total\_weight}$$

Be sure to use **`probs[chosen]` from `softmax(logits)` (the original probability)**, not the reshaped `filtered[chosen]`
(the source's `accept()` uses the cached `original_probs[pending_token_idx]`). `total_weight` increments by 1 each step
and decays geometrically, so $\bar p$ is a moving average of recent hit probabilities.

## Contract

Consistent with the collection: `filtered` is returned in the original order, and `chosen` comes from inverse-transform sampling on the original-order CDF; additionally, the two EMA states
are returned as the 3rd and 4th items, to be passed in on the next step.
