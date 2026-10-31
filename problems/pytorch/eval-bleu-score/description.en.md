# Evaluation: Corpus BLEU

## Problem

BLEU (Papineni et al., ACL 2002) ignores semantics entirely and only asks **what fraction of the n-grams in the system output can be found in a human reference**, then multiplies by a brevity penalty that punishes translations that are too short. All of the difficulty lives in three details: n-gram matches must be **clipped by how often they occur in the reference** (otherwise a single "the the the …" sentence saturates unigram precision), the per-order precisions enter through a **geometric mean** (any zero order zeroes out the whole score), and the brevity penalty's denominator is the **closest reference length per hypothesis**. To keep the metric reproducible, the inputs here are already segmented token-id sequences; no tokenizer is involved.

## Rules

- `hypotheses`: `list[list[int]]` — \(M\) candidate translations, each a list of token ids.
- `references`: `list[list[list[int]]]` — `references[i]` is the **reference set** of hypothesis \(i\) (at least one reference; 1–4 in practice), aligned one-to-one with the hypotheses.
- `max_n`: highest order, computing \(n = 1 \dots max\_n\) (default 4).
- `smoothing`: boolean switch for add-one smoothing (below).

Corpus-level statistics for each order \(n\):

1. **Denominator**: for each hypothesis enumerate all **contiguous** n-grams of \(n\) ids (counting repeats); the number of them, \(\max(0, |h|-n+1)\), accumulates into \(C_n = \sum_i \max(0, |h_i| - n + 1)\).
2. **Numerator (clipped matches)**: for each distinct n-gram \(g\) occurring in a hypothesis,
   \[ \text{clip}(g) = \min\big(\text{count}_{h}(g),\ \max_{j} \text{count}_{r_j}(g)\big) \]
   where \(\max_j\) runs over the **multiple references of that same hypothesis** (not over the corpus); accumulate into \(M_n\).
3. \(p_n = M_n / C_n\). If \(C_n = 0\) (the corpus contains no n-gram of that order at all, e.g. every hypothesis is shorter than \(n\)), define \(p_n = 0\).
4. **Smoothing**: when `smoothing` is true use Laplace add-one \(p_n = (M_n + 1)/(C_n + 1)\); orders with \(C_n = 0\) still stay \(0\) — "no n-gram of this order exists" is not the same as "precision of this order is zero" and add-one cannot rescue it. The returned precisions are the smoothed values that actually enter the geometric mean.
5. **Brevity penalty**: with \(c = \sum_i |h_i|\) and \(r = \sum_i \text{closest}_i\), where \(\text{closest}_i\) is the reference length whose difference to \(|h_i|\) is smallest, **ties going to the shorter one**:
   \[ \text{BP} = \begin{cases} 1, & c \ge r \\ e^{\,1 - r/c}, & c < r \end{cases} \]
   When \(c = 0\) the penalty is undefined; set BLEU \(= 0\) directly.
6. **Geometric mean**: \(\text{BLEU} = \text{BP}\cdot\exp\big(\frac{1}{max\_n}\sum_{n=1}^{max\_n}\ln p_n\big)\); if any \(p_n = 0\) then BLEU \(= 0\) (equivalently the \(1/max\_n\)-th root of the product with uniform weights).

Return the pair `(bleu, precisions)`: `bleu` is a Python float and `precisions` a float32 tensor of shape \((max\_n,)\) ordered \(p_1 \dots p_{max_n}\). Using `nltk` / `sacrebleu` / `torchtext` or any ready-made BLEU implementation is forbidden (they are not installed anyway).

## Example

```
hypotheses = [[1,2,3,4,5,6],          references = [[[1,2,3,4,5,6], [1,2,3,4,7,6]],
             [2,3,4,8]]                                          [[1,2,3,4,5]]]

n = 1:  C = 6 + 4 = 10,  M = 6 + 3 = 9    -> p1 = 0.9      (id 8 has no reference occurrence)
n = 2:  C = 5 + 3 = 8,   M = 5 + 2 = 7    -> p2 = 0.875    (bigram (4,8) misses)
n = 3:  C = 4 + 2 = 6,   M = 4 + 1 = 5    -> p3 = 0.8333333
n = 4:  C = 3 + 1 = 4,   M = 3 + 0 = 3    -> p4 = 0.75     (4-gram (2,3,4,8) misses)

c = 6 + 4 = 10
r = 6 + 5 = 11    (hypothesis 0 has length 6 and both references have length 6; hypothesis 1 has length 4, its closest reference has length 5)
c < r -> BP = exp(1 - 11/10) = 0.9048374

bleu = 0.9048374 * (0.9 * 0.875 * 0.8333333 * 0.75)^(1/4) = 0.7578848
precisions = [0.9, 0.875, 0.8333333, 0.75]

Same corpus with smoothing=True:
precisions = [0.9090909, 0.8888889, 0.8571429, 0.8],  bleu = 0.7806746
```

## Constraints

- CPU; token ids are non-negative integers in an arbitrary range (only equality is tested, no vocabulary mapping).
- Hypotheses and references may be **empty lists** (length 0); `references[i]` is guaranteed non-empty.
- **Degenerate properties**: every hypothesis identical to its reference \(\Rightarrow\) all \(p_n = 1\), \(c = r\), \(\text{BLEU} = 1\); some order with \(C_n = 0\) \(\Rightarrow\) \(p_n = 0\) and unsmoothed BLEU \(= 0\); duplicating a reference leaves the score unchanged (clipping uses a max, closest length uses a minimum distance); \(c > r\) \(\Rightarrow\) BP saturates at 1, so being longer than the reference earns no bonus.
- \(0 \le \text{BLEU} \le 1\).
