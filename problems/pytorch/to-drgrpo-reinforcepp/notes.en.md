# Solution Notes · Dr. GRPO and REINFORCE++

## Where the two biases come from

In the original GRPO objective each answer's contribution is divided by $|o_i|$ (length normalization) and the advantage is divided by the in-group std (difficulty normalization). The Dr. GRPO paper shows both steps leak "answer length" and "question difficulty" into the gradient:

- **Length bias**: dividing by $|o_i|$ gives each token of a long answer a smaller weight, effectively turning the token-level objective into "equal weight per answer" — the model has no reason to write longer or shorter and the optimization direction is distorted;
- **Difficulty bias**: dividing by the std amplifies the advantages of low-variance groups (almost all correct or all wrong), which are exactly the groups with the least information.

Dr. GRPO therefore replaces both by **constants**: the advantage only subtracts the group mean, and the loss divides by $B\cdot G\cdot T$. Note $T$ is the tensor width, not the valid length — so padding enters the denominator, and the more padding in the batch the more diluted the loss. That is not a bug but a direct consequence of "treat every position alike"; the problem pins it down with "appending a fully masked column shrinks the loss to 3/4".

## The whitening range of REINFORCE++

REINFORCE++ does not standardize per group; it whitens once over the **whole batch**: $\mu,\sigma$ come from all $B\cdot G$ rewards. All answers in a batch thus share one scale, and the relative difficulty between groups is preserved (unlike GRPO, which flattens each group to unit variance). Its loss still averages each answer over its own length and then averages over answers, so an extra padding column changes nothing — a clean contrast with Dr. GRPO.

One implementation caveat for $\sigma$: `std(unbiased=False)` of a constant reward tensor is not guaranteed to be exactly 0 in floating point, so the contract uses the test $\sigma\le10^{-12}$ and returns zeros for the whole batch in that case, keeping $0/0$ from injecting NaN into the loss.

## Clipping semantics of the surrogate

Both share $\min(\rho A,\ \bar\rho A)$. Since $A$ can be negative, this `min` does not always select the clipped value: for $A>0$ it caps an oversized $\rho$ at the upper bound (conservative), while for $A<0$ it keeps the more negative unclipped term (PPO's automatic one-sided clipping). The tests construct a negative-advantage sample with $\rho=e^{0.3}$, $\epsilon=0.01$ to verify the implementation does not just "take the clipped value symmetrically".

## Why merge the two into one problem

Seen separately, both are a few lines of normalization code; together, with identical inputs differing only in normalization, the outputs change proportionally or not at all — which is what makes the two choices "center vs whiten, and length vs constant denominator" visibly determine the gradient scale.
