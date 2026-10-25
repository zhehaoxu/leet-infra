# Evaluation: Shingle Contamination Audit

## Problem

Before training a language model you have to answer an awkward question: did the evaluation set leak into the training corpus? Once contaminated, the test score is inflated — the model did not "learn to reason", it "memorised the answers". A shingle contamination audit answers it with a reproducible overlap matrix: chop every document into a **set of its contiguous n-grams** (shingling, a technique dating from near-duplicate document detection, Broder 1997), then, pair by pair, measure "what fraction of this evaluation document's shingles shows up inside some training document".

The choice of ratio is the whole point. Jaccard \(\frac{|A\cap B|}{|A\cup B|}\) is symmetric, and once a training document is long (real corpora run to tens of thousands of tokens), the union blows up on the training side, so even a **verbatim-contained** evaluation document is left with a number close to 0. Containment \(\frac{|A_i\cap B_j|}{|A_i|}\) pins the denominator to the **evaluation side**: it asks "how much of this evaluation document is covered by that training document", and is never diluted by the training document's length — which is exactly why LLM leakage audits reach for n-gram containment instead of Jaccard. This problem computes both matrices and flags the evaluation documents that cross a threshold. To keep judging deterministic and to sidestep any tokenisation difference, the inputs are already-cut token-id sequences; no tokenizer and no de-duplication library is touched.

## Rules

- `eval_docs`: `list[list[int]]` — \(M_{\text{eval}}\) evaluation documents to audit; each is a sequence of non-negative integer token ids (already segmented, no tokenizer needed).
- `train_docs`: `list[list[int]]` — \(M_{\text{train}}\) training-corpus documents, same form.
- `n`: a positive integer, the shingle length (contiguous id window).
- `threshold`: a float, the hit threshold.

**Shingle set**: the shingle set \(S(d)\) of a document \(d\) is the collection of all its **contiguous \(n\) ids** (overlapping windows), **de-duplicated** — the same \(n\)-gram appearing several times counts once. If \(|d| < n\) no window can be taken and \(S(d) = \varnothing\). Write \(A_i = S(\text{eval\_docs}[i])\) for evaluation document \(i\) and \(B_j = S(\text{train\_docs}[j])\) for training document \(j\).

For every pair \((i, j)\) (\(0 \le i < M_{\text{eval}},\ 0 \le j < M_{\text{train}}\)):

- **Containment** (denominator on the evaluation side, asymmetric):
  \[ \text{contam}[i,j] = \frac{|A_i \cap B_j|}{|A_i|} \]
- **Jaccard** (symmetric, denominator is the union):
  \[ \text{jac}[i,j] = \frac{|A_i \cap B_j|}{|A_i \cup B_j|}, \qquad |A_i \cup B_j| = |A_i| + |B_j| - |A_i \cap B_j| \]
- **Empty-set sentinel**: when \(|A_i| = 0\) the whole row is recorded as \(0.0\) (a \(0/0\) never arises), and an empty union is likewise \(0.0\). So any evaluation document shorter than \(n\) has both its contam row and its jac row all zeros.

**Hit list `flagged`**: an **ascending** list of evaluation-document indices; document \(i\) is included iff all three hold—

1. \(|A_i| > 0\) (its shingle set is non-empty);
2. the training corpus is non-empty (\(M_{\text{train}} > 0\));
3. \(\max_j \text{contam}[i,j] \ge \text{threshold}\) (**equality counts as a hit**).

**Return** the triple `(contam, jac, flagged)`: `contam` and `jac` are float32 tensors of shape \((M_{\text{eval}}, M_{\text{train}})\), and `flagged` is an int64 tensor. If either side is empty the matrices degrade to shape \((M_{\text{eval}}, 0)\), \((0, M_{\text{train}})\) or \((0, 0)\) accordingly, and `flagged` is empty.

Using any off-the-shelf de-duplication / near-duplicate / text-similarity library is **forbidden**; shingle sets must be built by hand with windows and set operations.

## Example

```
eval_docs = [
    [1, 2, 3, 6, 4],       # e0 -> A0 = {(1,2,3),(2,3,6),(3,6,4)},            |A0| = 3
    [1, 5, 3, 6, 4],       # e1 -> A1 = {(1,5,3),(5,3,6),(3,6,4)},            |A1| = 3
    [1, 2],                # e2 -> length 2 < n=3, A2 = empty
]
train_docs = [
    [1, 2, 3, 6, 4],            # t0 -> B0 = {(1,2,3),(2,3,6),(3,6,4)},                                          |B0| = 3
    [1, 2, 3, 6, 4, 7, 8, 9],   # t1 -> {(1,2,3),(2,3,6),(3,6,4),(6,4,7),(4,7,8),(7,8,9)},                       |B1| = 6
    [1, 5, 3, 6, 4, 2, 9],      # t2 -> {(1,5,3),(5,3,6),(3,6,4),(6,4,2),(4,2,9)},                               |B2| = 5
]
n = 3,  threshold = 0.6

containment (denominator |A_i| always on the eval side; count the intersection, divide by the eval set size):
  e0: |A0∩B0|=3 -> 3/3=1.0   |A0∩B1|=3 -> 3/3=1.0   |A0∩B2|=1 -> 1/3=0.3333333
  e1: |A1∩B0|=1 -> 1/3       |A1∩B1|=1 -> 1/3       |A1∩B2|=3 -> 3/3=1.0
  e2: |A2|=0 -> whole row 0.0, 0.0, 0.0
contam = [[1.0,        1.0,        0.3333333],
          [0.3333333,  0.3333333,  1.0],
          [0.0,        0.0,        0.0]]

Jaccard (denominator |A_i ∪ B_j| = |A_i| + |B_j| - |A_i∩B_j|):
  e0: inter3 union3 -> 1.0   inter3 union6 -> 0.5   inter1 union7 -> 0.14285715
  e1: inter1 union5 -> 0.2   inter1 union8 -> 0.125 inter3 union5 -> 0.6
  e2: all 0.0
jac = [[1.0,   0.5,   0.14285715],
       [0.2,   0.125, 0.6],
       [0.0,   0.0,   0.0]]

flagged (max_j contam >= 0.6 per row; the empty e2 is never included):
  e0 max=1.0 ✓    e1 max=1.0 ✓    e2 skipped
flagged = [0, 1]
```

Look closely at the e0-vs-t1 cell: containment is \(1.0\) (all three shingles of the evaluation document are covered by the longer t1), yet Jaccard is only \(0.5\) (the union is inflated to 6 by t1's three extra shingles). The same verbatim match gives numbers twice apart — that gap is exactly the difference between pinning the denominator to the evaluation side versus pinning it to the union.

## Constraints

- Judge environment: CPU only; `torch` / `numpy` available offline, no network; `nltk`, MinHash/LSH, `datasketch`, `text-dedup` and any other off-the-shelf de-duplication/similarity library are **forbidden** (and not installed); shingle sets must be written by hand.
- Token ids are non-negative integers; this problem only compares them for equality and does no vocabulary mapping; \(n \ge 1\).
- \(0 \le \text{contam} \le 1\) and \(0 \le \text{jac} \le 1\), with \(\text{jac}[i,j] \le \text{contam}[i,j]\) elementwise (since \(|A_i \cup B_j| \ge |A_i|\)).
- **Degenerate behaviour**:
  - An evaluation document shorter than \(n\) ⇒ \(A_i = \varnothing\) ⇒ its contam and jac rows are all 0, and it never enters `flagged` even when `threshold = 0`.
  - Empty training side ⇒ matrix shape \((M_{\text{eval}}, 0)\) and `flagged` is empty (even when `threshold = 0`); an empty evaluation side likewise gives \((0, M_{\text{train}})\).
  - Evaluation corpus identical to the training corpus, document by document ⇒ the diagonal has contam = jac = 1, hit at `threshold = 1.0` (boundary inclusive).
  - Fully disjoint vocabularies ⇒ both matrices are all 0; `threshold > 0` flags nobody, `threshold = 0` flags everyone (except empty-set rows).
  - Set semantics ⇒ containment is always \(\le 1\): a repeated n-gram counts once, so \([1,1,1]\) at \(n=2\) yields only \(\{(1,1)\}\).
  - `threshold > 1` ⇒ `flagged` is necessarily empty (containment's upper bound is 1).
