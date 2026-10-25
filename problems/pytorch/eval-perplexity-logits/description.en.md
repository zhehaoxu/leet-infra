# Evaluation: Perplexity from Logits

## Problem

Perplexity (PPL) is the most basic intrinsic metric for a language model: take the geometric mean of the probability the model assigns to the **true next token** at every position and invert it, which equals \(\exp(\overline{\text{NLL}})\). Its interpretability comes from "effective branching factor" — \(\text{PPL}=1\) means the model never hesitates, \(\text{PPL}=V\) (the vocabulary size) means it is no better than guessing uniformly over the vocabulary. Mikolov's RNNLM report (2012) and HuggingFace `evaluate`'s `perplexity` both use this definition.

## Rules

- `logits`: float32, shape \((B, L, V)\) — the model's unnormalised scores for "the next token" at each input position.
- `labels`: int64, shape \((B, L)\) — target token ids, as long as the input; every position that is not supervised (prompt segment, padding) is written as `ignore_index` (default \(-100\)).
- **Right-shift alignment**: the logits at position \(i\) predict label \(i+1\), so the paired set is the first \(L-1\) logit positions against the last \(L-1\) labels; the logits at the final position have no target and are dropped; `labels[..., 0]` never corresponds to a prediction.
- Per-position loss: \(\text{nll}_i = -\log\mathrm{softmax}(\text{logits}_i)[\text{label}_i]\), with a numerically stable log-softmax (subtract the row maximum first).
- The denominator counts **valid positions only**: let \(n\) be the number of pairs whose label differs from `ignore_index`, then PPL \(=\exp\big(\frac1n \sum \text{nll}\big)\) — a micro average over all valid tokens; averaging per-sample losses and then averaging over samples is wrong.
- If \(n = 0\) (prompt-only, all padding, or \(L=1\) so the shift leaves no pair), return the sentinel **\(-1.0\)**: on a non-empty valid set PPL is always \(\ge 1\), so \(-1\) can never collide with a real value.
- Return a **0-dimensional float32 tensor**. Ready-made evaluation libraries are forbidden.

## Example

```
logits[0] = [[ 2.0,  0.5, -1.0],
             [ 0.0,  1.0,  3.0],     labels[0] = [-100, 0, 2]
             [ 0.2,  0.1,  0.3]]     shifted labels = [0, 2]
                                     (labels[0][0] = -100 is prompt and falls out of the shift
                                      window; the logits at position 2 have no next label)

logits[1] = [[ 1.0,  1.0,  1.0],
             [-2.0,  0.5,  0.5],     labels[1] = [1, -100, 1]
             [ 0.0,  0.0,  4.0]]     shifted labels = [-100, 1]
                                     (exactly one valid position)

3 valid pairs:
(row 0, position 0 -> id 0) nll = 0.241311
(row 0, position 1 -> id 2) nll = 0.169846
(row 1, position 1 -> id 1) nll = 0.733370
mean_nll = 1.144527 / 3 = 0.381509
ppl = exp(0.381509) = 1.4644929
```

## Constraints

- CPU float32; \(V\ge 2\), \(B, L \ge 1\); every valid label satisfies \(0 \le \text{id} < V\).
- On a non-empty valid set \(\text{PPL} \in [1, +\infty)\).
- **Degenerate properties**: if every row of logits is constant (uniform softmax) then \(\text{PPL}=V\); if each valid position puts all its mass on the correct id then \(\text{PPL}\to 1\); if `labels` is entirely `ignore_index`, or \(L=1\), return \(-1.0\).
- Only shifted positions are counted: forgetting the shift, or shifting the other way, breaks the three per-position nlls above immediately.
