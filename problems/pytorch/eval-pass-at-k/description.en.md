# Evaluation: The Unbiased pass@k Estimator

## Problem

For each code problem a model typically produces \(n\) candidate programs which are run against unit tests, yielding \(c\) passes. What should be reported is **not** \((1-(1-c/n)^k)\), which treats the samples as independent draws, but the **unbiased estimator** of Chen et al. (*Evaluating Large Language Models Trained on Code*, arXiv:2107.03374, Codex/HumanEval): drawing \(k\) samples **without replacement** from the \(n\), the probability that at least one passes is exactly

\[ \mathbb{E}\big[\text{pass@}k\big] = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \]

The denominator counts all \(k\)-subsets, the numerator those taken entirely from the failing samples. The formula degenerates to the pass rate \(c/n\) at \(k=1\), and at \(k=n\) it says "as long as one correct sample exists the score is 1" — it is about **subset coverage**, not \(k\) independent trials. HumanEval, MBPP and EvalPlus all report this quantity.

## Rules

- Inputs are Python ints: \(n\) (number of samples), \(c\) (how many pass, \(0 \le c \le n\)), \(k\) (how many are drawn, \(1 \le k \le n\)).
- The ratio \(\binom{n-c}{k}/\binom{n}{k}\) must be formed from **exact integer binomial coefficients** (`math.comb` or an equivalent hand-written version), converting to floating point **only at the last step**; intermediate floating-point factorials or `math.gamma` lose the significant digits once \(n\) reaches a few hundred.
- \(\binom{a}{b} = 0\) whenever \(b > a\) (so \(\text{pass@}k = 1\) in that case); do not raise, do not return nan.
- The approximation \(1-(1-c/n)^k\) is forbidden, as are evaluation libraries.
- Return a **Python float** in \([0, 1]\).

## Example

```
n = 10, c = 2, k = 3
C(n-c, k) = C(8, 3) = 56
C(n, k)   = C(10, 3) = 120
pass@3 = 1 - 56/120 = 0.5333333333

for contrast: the common wrong formula 1 - (1 - 2/10)^3 = 0.488, biased and systematically low whenever k > 1
```

## Constraints

- CPU; pure integer/float arithmetic, no randomness at all.
- **Degenerate properties** (all follow from binomial identities):
  - \(c = 0 \Rightarrow \text{pass@}k = 0\);
  - \(k = 1 \Rightarrow \text{pass@}k = c/n\) (this coincides with the naive formula and anchors unbiasedness);
  - \(k = n \Rightarrow 0\) if \(c = 0\), otherwise \(1\) (taking every sample must cover a correct solution);
  - \(c = n \Rightarrow\) always \(1\);
  - \(n - c < k \Rightarrow\) always \(1\) (there are not enough failing samples to fill a \(k\)-subset);
  - for fixed \(n, c\), pass@k is **non-decreasing in \(k\)**.
- Range: \(0 \le \text{pass@}k \le 1\).
