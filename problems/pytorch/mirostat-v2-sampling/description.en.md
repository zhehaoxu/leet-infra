# Mirostat v2 Sampling

## Problem

top-k / top-p truncate with a fixed threshold, but "how many candidates to keep" should really vary with the model's current **degree of uncertainty**. **Mirostat v2** (Mirsky et al.) sets the sampling objective as **controlling perplexity** near a configured value $\tau$: it maintains an internal threshold $\mu$, each step cuts the long-tail tokens that are "more surprising than $\mu$", and then **adjusts $\mu$** based on the perplexity feedback of the token actually drawn this step — if the outcome is too surprising (perplexity above $\tau$), tighten $\mu$ and cut harder next time; if it is too bland, relax $\mu$ and allow more candidates. This is a closed-loop controller with **state**.

> To make grading **deterministic**, the uniform random number `u` is passed in as an explicit input; the state $\mu$ is both an input and part of the returned output.

## Rules

- Implement `solve(logits, tau, eta, mu, u)`: `logits` has shape `(..., V)`; `tau` (target perplexity), `eta` (learning rate) and `mu` (current threshold, a float) are scalar parameters; `u` is one random number per row with value in $[0,1)$ (shape `(...,)` or a scalar).
- Steps (row by row):
  1. `probs = softmax(logits)`.
  2. Sort by probability in **descending** order, and let each candidate's surprisal be $s_i = -\log_2 p_i$. Find the **first** position $k$ with $s_i > mu$ and keep only the first $k$ of the descending prefix (dropping the more surprising tail after it); if even the first element has $s_0>mu$, keep at least 1 ($size=\max(k,1)$); if nothing exceeds the threshold, keep everything.
  3. Renormalize the kept set → `filtered` (mapped back to the original order, same shape as `probs`, each row summing to 1).
  4. Use `u` to sample `chosen` by inverse transform over the CDF of `filtered` (original order).
  5. Observe the perplexity $\text{obs} = -\log_2(\text{filtered}[\text{chosen}])$ and update
     $$mu_{\text{new}} = mu - \eta\,(\text{obs} - \tau)$$
- Return `(filtered, chosen, mu_new)`.

## Examples

```
Input: logits = [2, 1, 0.5, -1, -2, -3], tau = 5.0, eta = 0.1, mu = 4.0, u = 0.5
  probs = [0.600294, 0.220836, 0.133944, 0.029887, 0.010995, 0.004045]
  surprisal(-log2 p) = [0.736, 2.179, 2.900, 5.064, 6.507, 7.950]
  the first > mu(4.0) is index 3 → keep the first 3 in descending order (which are indices 0,1,2), normalize
  filtered = [0.628532, 0.231224, 0.140244, 0, 0, 0]
  CDF = [0.628532, 0.859756, 1.0], u = 0.5 → chosen = 0
  obs = -log2(0.628532) = 0.670434; mu_new = 4.0 - 0.1*(0.670434 - 5.0) = 4.433006
Output: filtered = [0.628532, 0.231224, 0.140244, 0, 0, 0], chosen = 0, mu_new = 4.433006
```

## Constraints

- float32 input; `tau > 0`, `eta > 0`; `u ∈ [0,1)`
- Each row of `filtered` sums to 1, and discarded positions are exactly 0; `chosen` must fall inside the kept set (≥1 element)
- `mu_new` has shape `(...,)` (matching the number of rows; a scalar tensor for a single row)
- Feedback direction: the blander the drawn token (obs < tau) → mu rises and the next step relaxes; the more surprising → mu falls and it tightens
