# Speculative Decoding: Closed-Form Speedup from Acceptance Rate and Optimal Gamma

## Problem

The payoff of speculative decoding reduces to two quantities: **how many tokens a step emits in expectation** and **how much time that step costs**. Two common misjudgments—"an 80% acceptance rate means 80% faster" and "longer drafts are always better"—are both refuted by the formulas below.

Normalize the target step latency to 1 and let a draft token cost \(c\) (`cost_ratio`). One speculative step generates \(\gamma\) draft tokens and runs one target forward, so the step latency is \(1 + \gamma c\).

The expected token count comes from the per-position acceptance rates \(\alpha_1,\dots,\alpha_\gamma\): after position \(\gamma\) there is still one "free" bonus token (the target samples it itself), hence

\[
e(\gamma) = 1 + \sum_{i=1}^{\gamma}\ \prod_{j=1}^{i}\alpha_j
\]

The first term is the token you always get; the \(i\)-th term is the probability that the first \(i\) drafts are all accepted—once a position is rejected, nothing beyond it can be emitted.

The speedup spreads the extra tokens over the extra time:

\[
s(\gamma) = \frac{e(\gamma)}{1 + \gamma c},\qquad s(0) = 1
\]

\(\gamma = 0\) means no speculation (a target forward only), giving the baseline speedup of 1. When the per-position acceptance rate is a constant \(\alpha\), \(e(\gamma)\) degenerates to the geometric series \(\frac{1-\alpha^{\gamma+1}}{1-\alpha}\).

The real knob is \(\gamma\): a longer draft raises the numerator (with diminishing returns, since \(\prod\alpha_j\) decays exponentially) and raises the denominator linearly. So \(s(\gamma)\) typically has an interior optimum instead of "longer is better".

## Task

Implement `solve(alphas, cost_ratio)`:

- Input: `alphas` (`(G,)` float32, per-position acceptance rates in \([0,1]\); the upper bound of \(\gamma\) is \(G\)); `cost_ratio` (float, \(\ge 0\)).
- Outputs:
  - `expected` (`(G+1,)` float32): \(e(\gamma)\) for \(\gamma = 0..G\);
  - `speedup` (`(G+1,)` float32): the corresponding \(s(\gamma)\);
  - `best_gamma` (int64 scalar): argmax of \(s\), **ties to the smallest \(\gamma\)**;
  - `best_speedup` (float32 scalar): that maximum.
- Compute internally in float64 and cast to float32; raise `ValueError` when `alphas` is empty, out of range, or `cost_ratio < 0`.

## Example

```
alphas = [0.8, 0.6, 0.5], cost_ratio = 0.1

γ = 0: e = 1                                  s = 1 / 1.0  = 1.0000
γ = 1: e = 1 + 0.8            = 1.8           s = 1.8 / 1.1 = 1.6364
γ = 2: e = 1.8 + 0.8*0.6      = 2.28          s = 2.28/ 1.2 = 1.9000
γ = 3: e = 2.28 + 0.8*0.6*0.5 = 2.52          s = 2.52/ 1.3 = 1.9385

expected    = [1.0, 1.8, 2.28, 2.52]
speedup     = [1.0, 1.6364, 1.9, 1.9385]
best_gamma  = 3
best_speedup= 1.9385
```

Note that going from \(\gamma=2\) to \(\gamma=3\) emits only 0.24 extra tokens while costing 0.1 extra time units—the gain is nearly eaten by latency.

## Constraints

- **The denominator is \(1+\gamma c\)**: the \(\gamma\) draft tokens are generated one at a time (autoregressive), so they cannot be billed as a single forward.
- **The bonus token always counts**: \(e(0)=1\) and \(s(0)=1\). Writing \(e\) as "number of accepted tokens" and dropping the bonus biases every speedup downward.
- **Ties go to the smallest \(\gamma\)**: with `cost_ratio = 1` and full acceptance every \(\gamma\) has speedup 1, and the answer should be 0 (no speculation), not the largest \(\gamma\).
- **Per-position rates are not one \(\alpha\)**: early positions are usually accepted more often; substituting a single averaged \(\alpha\) overestimates long drafts.
- Do not stop at the first decrease ("unimodal, exit early"): with noisy acceptance rates \(s\) is not guaranteed unimodal.
