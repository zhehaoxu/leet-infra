# Tokenization: Unigram Optimal Subword Segmentation (Viterbi)

## Problem

SentencePiece's **Unigram** language model assigns each subword \(x\) in the vocabulary an occurrence probability \(p(x)\). The probability of cutting a string into a sequence of subwords \(S=(x_1,\ldots,x_k)\) is the product of the individual subword probabilities, which becomes a sum after taking logs. Encoding is thus a shortest-path search on the grid of "all possible segmentations": **pick the segmentation with the highest total log probability**

\[ \hat{S} = \arg\max_{S} \sum_{x \in S} \log p(x) \]

This is classic dynamic programming (Viterbi), \(O(nL)\) (\(n\) the string length, \(L\) the longest subword), and finds the global optimum better than "greedy longest match".

## Rules

Implement `solve(text, vocab, log_probs)`, returning a 1-D int64 tensor whose elements are the **indices of the selected subwords in `vocab`** (concatenated left to right they equal `text` exactly).

- `text`: str, the string to segment.
- `vocab`: list[str], the subword vocabulary (may contain single characters). `log_probs`: list[float], the same length as `vocab`, giving each subword's log probability (\(\le 0\)).
- Use Viterbi / DP to pick, among all segmentations whose subwords "connect head to tail and fully cover `text`", the one maximizing \(\sum \log p\), and return its sequence of subword indices.
- If no segmentation can fully cover `text` (some character cannot be spelled from the vocabulary), raise `ValueError`.

## Examples

`text="unigram"`, `vocab=["u","n","i","g","r","a","m","un","uni","gram","nig","am"]`, with probabilities
`[.05,.05,.08,.05,.05,.09,.05,.1,.5,.4,.01,.02]` converted to natural logs as `log_probs`:

```
Best segmentation = ["uni", "gram"]  ->  return the index tensor [8, 9]
("uni" has probability 0.5 and "gram" 0.4; the product 0.2 far exceeds per-character or any other split)
```

`text="abc"`, `vocab=["a","b","c","ab","bc","abc","ac"]`, probabilities `[.01,.01,.01,.05,.02,.6,.03]`:

```
Best segmentation = ["abc"]  ->  return the index tensor [5]
```

## Constraints

- The judge cases guarantee the **optimal segmentation is unique** (no tied maximal log probability), so the return value does not depend on tie-breaking rules.
- Subwords in `vocab` are non-empty; the same segmentation is returned as `vocab` indices.
- Raise `ValueError` when full coverage is impossible (assert negative cases with try/except, do not rely on pytest).
- Pure deterministic string + floating-point DP, CPU, no randomness.
