# Solution Notes · A2S Adaptive Sparsity

## Two decoupled mechanisms

The problem splits "where do candidates come from" and "how many do we take" into two steps, which is the key to understanding A2S:

- **Where candidates come from**: to skip the full scoring pass of the current step, candidates can only be the previous step's top-\(k\). Attention is strongly correlated between adjacent decoding steps, so this prior is usually accurate.
- **How many to take**: once candidates are fixed, order them by the **current** scores and accumulate mass up to \(p\), yielding an adaptive \(n\). Steps with concentrated attention get a small \(n\); dispersed steps push \(n\) to the candidate cap.

Get both right and the result is "previous step's indices plus a handful of current comparisons". Get either wrong and it degenerates: all-column candidate selection saves no scoring, and previous-step ordering collapses to a fixed top-k.

## Why n must be adaptive

A fixed n wastes in both directions. When attention concentrates on a few tokens, the extra keys contribute almost nothing and burn bandwidth; when attention is dispersed, a fixed n covers only half the mass and accuracy drops. The mass threshold \(p\) turns "is that enough" into a measurable criterion: stop once the accumulated fraction reaches \(p\). It also makes \(p\) a monotone knob—larger \(p\) means larger \(n\)—so the accuracy/latency trade-off becomes a one-dimensional search.

## The price of approximation

Candidates from the previous step inevitably miss things: a step whose attention suddenly lands on a token that was not selected before (new entity, new instruction, abrupt format change) cannot see it at all. That is not a bug but the inherent cost of skipping a full scoring pass; in production it is covered by periodic full refreshes or by always keeping the sink tokens.

The `pattern_reuse_limits_recall` check pins this down: if the highest-scoring token of the current step is outside the previous top-\(k\), it must not be selected.

## Boundaries and ties

- Zero total mass (extremely sparse or all-zero input): the cumulative sum stays 0, so `>= pT` already holds at \(m=1\) and \(n=1\). This encodes "at least one" and avoids division by zero.
- Early steps are limited by the number of valid columns: right after decoding starts, the valid count may be below \(k\), so the candidate count is the valid count and the bound on `n` shrinks accordingly.
- Every ordering is "descending, ties to the smaller index", consistent with the library; comparisons keep a \(10^{-12}\) tolerance so accumulated rounding cannot flip a boundary case.

## Complexity

\(O(k\log k)\) per step (sorting within the candidates) instead of \(O(L\log L)\) over all columns—that is the main saving of sparsification. The mass accumulation is \(O(n)\le O(k)\). Total time \(O(Sk\log k)\), space \(O(SL)\) for the output mask.

## Related problems

- `kv-expected-attention`: scores keys with a Gaussian expectation and takes top-k—a "scorer" perspective, whereas this problem changes only the **candidate set and budget**.
- `kv-tova-head-mean`: fixed top-k, budget does not change per step.
- `kv-h2o-heavy-hitters`: also emits a per-step retained set, but from cumulative scores rather than the previous step's pattern.
