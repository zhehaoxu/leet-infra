# Entropy-Bounded Unmasking: Low-Entropy Positions Commit First

## Problem Statement

In discrete diffusion — especially the uniform-corruption family, which has no MASK token and simply replaces corrupted positions with uniformly random tokens — how many positions to commit per step decides both the step count and the quality. Commit too few and steps are wasted; commit too many and the tokens fixed together are strongly correlated, so decoding collapses into independent per-position argmax.

Entropy-bounded unmasking works as follows: each step computes every position's distribution entropy at an annealed temperature, sorts positions by entropy **ascending**, and accepts only the prefix whose joint entropy stays under a threshold. Accepted positions keep this step's sampled candidate token, the rest are renoised with uniformly random tokens, and the next round tries again.

This problem covers a **single step**: given the temperature parameters, the current step index, the model logits, and the candidate / random tokens already drawn by the caller, output this step's token block, acceptance mask and per-position entropy.

## Rules

Let the number of steps be \(N\), the current step index be \(s\) (\(0\le s<N\)), and the entropy bound be \(B\).

1. **Temperature annealing** (hottest at the first step, sharpest at the end):
   \[
   \tau=t_{\min}+(t_{\max}-t_{\min})\cdot\frac{N-s}{N}
   \]
   Note that at the last step \(s=N-1\) the temperature is \(t_{\min}+(t_{\max}-t_{\min})/N\) — it only approaches \(t_{\min}\), never equals it.
2. **Scale the logits and take entropies**: \(z=\text{logits}/\tau\), and per position
   \[
   p=\mathrm{softmax}(z),\qquad H=-\sum_v p_v\log p_v
   \]
   The candidates are drawn from the **same** scaled distribution.
3. **Sort ascending by entropy** (ties broken by smaller index) and take the prefix sums \(\text{cum}\) of the sorted sequence.
4. **The acceptance rule uses the sum that excludes the position itself**:
   \[
   \text{accepted}_k \iff \big(\text{cum}_k-H_k\big)\le B
   \]
   i.e. a position is accepted exactly when the entropies of all **strictly lower-entropy** positions sum to at most \(B\). Accepted positions therefore always form a lowest-entropy prefix, and \(B=0\) accepts only the single lowest-entropy position.
5. **Scatter back** to the original order and compose: `prev_sample = where(accepted, sampled_tokens, random_tokens)`.

Return `(prev_sample, accepted, token_entropy)`: `int64`, `bool` and `float32` respectively, all shaped `(batch, block_length)`; `token_entropy` is the entropy after temperature scaling.

## Example

`logits` of shape `(1, 3, 3)`:

```
position 0: [2, 0, -1]      position 1: [0, 0, 0]      position 2: [1, 1, 0]
sampled_tokens = [[1, 1, 0]]      random_tokens = [[2, 2, 1]]
timestep = 0, num_inference_steps = 4, t_max = 0.8, t_min = 0.4
```

The temperature is \(0.4+0.4\times 1=0.8\) and the three entropies are

```
entropy = [0.365770, 1.098612, 0.983653]
```

- \(B=1.0\): sorted order is position 0 (0.3658), position 2 (0.9837), position 1 (1.0986). The prefix sums for position 0 and position 2 are 0 and 0.3658, both under 1.0; for position 1 it is 1.3494, over the bound. Hence `accepted = [True, False, True]` and `prev_sample = [1, 2, 0]` (position 1 takes the random token 2).
- \(B=0.2\) (the default 0.1 gives the same result): only position 0 is accepted, `prev_sample = [1, 2, 1]`.
- \(B=10\): everything is accepted, `prev_sample = [1, 1, 0]`.

## Constraints

- `logits` must be a 3-D float tensor `(batch, block_length, vocab_size)` with all three dimensions \(\ge 1\) and no nan / inf;
- `sampled_tokens` and `random_tokens` must be `int32` / `int64` tensors shaped exactly `(batch, block_length)` with values in \([0,\text{vocab\_size})\);
- `timestep` is a Python integer (booleans rejected) in \([0,N)\); `num_inference_steps` is a positive integer;
- `entropy_bound` must be a non-negative finite real; `t_max` and `t_min` must be positive finite reals (a zero temperature divides by zero);
- Raise `ValueError` when any of the above fails;
- Properties: `accepted` is always a prefix in ascending-entropy order; larger bounds imply supersets of accepted positions; \(B=0\) accepts only the minimum-entropy position; identically distributed positions have equal entropy and ties go to the smaller index; `prev_sample` equals `sampled_tokens` where accepted and `random_tokens` elsewhere.
