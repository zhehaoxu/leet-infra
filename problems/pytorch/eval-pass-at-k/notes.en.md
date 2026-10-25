# In Depth: The Unbiased pass@k Estimator

## Origin
Chen, Tworek, Jun, Yuan, Pinto, Kaplan et al., *Evaluating Large Language Models Trained on Code* (arXiv:2107.03374), Eq. (1):

\[ \text{pass@}k \;:=\; \mathbb{E}_{C\sim \text{Unif}(\mathcal C_k)}\Big[\,1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\,\Big] \]

where \(\mathcal C_k\) is the set of all \(k\)-subsets of the \(n\) samples. The paper explicitly notes that plugging \(p = c/n\) into \(1-(1-p)^k\) is **biased**, and that the bias is not negligible at the sample sizes actually used (\(n=200\) in HumanEval). The reference implementation is `human_eval.compute_pass_at_k`; EvalPlus, BigCodeBench and OpenCompass all reuse this estimator.

## Derivation of unbiasedness
Let \(X\) be the number of passing programs among \(k\) samples drawn without replacement from \(n\), of which \(c\) pass. Then \(X\) is hypergeometric:
\[ \Pr[X = j] = \frac{\binom{c}{j}\binom{n-c}{k-j}}{\binom{n}{k}}, \qquad j = \max(0, k-(n-c)) \dots \min(c, k) \]
Hence
\[ \Pr[X \ge 1] = 1 - \Pr[X = 0] = 1 - \frac{\binom{c}{0}\binom{n-c}{k}}{\binom{n}{k}} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \]
because the only subsets with no hit are those drawn entirely from the \(n-c\) failures. The naive formula replaces the hypergeometric by a binomial model (independent draws with replacement, each passing with probability \(c/n\)):
\[ \frac{\binom{n-c}{k}}{\binom{n}{k}} = \prod_{i=0}^{k-1}\frac{n-c-i}{n-i} \;\le\; \Big(\frac{n-c}{n}\Big)^{k} = (1-c/n)^k \]
(the bound \(b(a-i) \le a(b-i)\) for \(a \le b\) is equivalent to \(ai \le bi\), true; so subtracting \(i\) from numerator and denominator drags a proper fraction further down). Taking complements reverses the inequality: \(1-(1-c/n)^k \le\) the true value, i.e. the naive version is **systematically low**, increasingly so as \(k\) grows — at \(n=10, c=2, k=3\): \(0.488\) against the correct \(0.5333\). At \(k=1\) both equal \(c/n\), which is the anchor of unbiasedness.

## Core points and pitfalls
1. **pass@1 is the pass rate \(c/n\)**: writing it as "did at least one candidate pass" (a 0/1 verdict) only coincides with the truth at \(k=n\); for other \(k\) it is a different quantity, and it is the most common mistake here.
2. **Exact integer ratio**: \(\binom{n-c}{k}\) and \(\binom{n}{k}\) are integers, and Python's `/` on two big integers performs true division with correct rounding; but **converting `comb` to float first** loses digits once \(n\) reaches a few hundred (`comb(200, 10) ≈ 2.2e16` is already past the \(2^{53}\) exact-integer range of float64). Factorials divided through `math.gamma` are worse.
3. **The \(\binom{a}{b}\) convention for \(b > a\)**: when \(k > n-c\) the numerator is 0 and pass@k is 1. That is both the combinatorial fact ("not enough failures to fill the subset") and the branch most implementations forget — returning a negative number or raising `ValueError` is wrong (`math.comb` already returns 0 there, which is exactly usable).
4. **Do not average across problems inside the estimator**: pass@k is a **per-problem** estimator; aggregating several problems means computing pass@k per problem and then averaging over problems (the standard macro view in code evaluation). This problem only produces the single-problem number. Compare with `eval-token-accuracy`: there the averaging unit inside a problem is a *valid token*, here across problems it is a *problem* — the same micro/macro distinction mirrored.
5. **Monotonicity**: pass@k is non-decreasing in \(k\) and pass@\(n = [c > 0]\). A non-monotone curve almost always means \(n-c\) and \(n\) were swapped between numerator and denominator.
6. **\(c\) counts distinct passing samples**, not "the position of the first pass" and not a rate: the same test suite can be passed by many candidates, and \(c\) must be that count.
7. The result lives in \([0,1]\): at \(c=0\) numerator and denominator are equal, so the value must be exactly 0 and never \(-10^{-16}\) from an unlucky evaluation order.

## Neighbouring metrics
- Versus the generation-side problems (`top-k-sampling`, `min-p-sampling`, …): those decide **how** the \(n\) candidates are produced; this one only turns their binary test outcomes into a metric.
- Versus `eval-ndcg-at-k`: both carry "at k", but in nDCG the \(k\) is a **truncation depth** (order matters, it is a prefix of a permutation), while here \(k\) is a **subset size** (order irrelevant, combinations not permutations). Different mathematical objects.

## Verification approach
The reference solution makes two `math.comb` calls and one true division. The judge's `_oracle` comes in two independent flavours: `_oracle_bruteforce` enumerates all \(\binom{n}{k}\) subsets with `itertools.combinations` and counts the fraction containing at least one correct index — the definition itself, used exhaustively for \(n \le 7\); `_oracle_product` evaluates \(\prod_i (n-c-i)/(n-i)\) (no binomials, no enumeration). Additional assertions cover the five identities (\(c=0\), \(k=1\), \(k=n\), \(c=n\), \(n-c<k\)), monotonicity in \(k\), the exact rational value 0.4 at \(n=5, c=1, k=2\), and the contrast against the biased \(1-(1-c/n)^k\). Float comparisons use `allclose(atol=1e-5, rtol=1e-5)` on float64 tensors; the exact 0/1 identities use `==`.

## Complexity
`math.comb` uses integer arithmetic, roughly \(O(k)\) big-integer multiplications plus one true division; the enumeration oracle costs \(O\big(\binom{n}{k}\big)\) and is therefore only run on small \(n\).

## Practical notes
The choice of \(n\) and \(k\) determines comparability: HumanEval uses \(n=200\) with \(k \in \{1, 10, 100\}\), and mixing pass@1 and pass@100 across the rows of one table is the classic chart error. pass@k only answers "can it be done", not "how reliably" — read it next to an intrinsic metric such as `eval-perplexity-logits`.
