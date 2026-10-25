# Evaluation: nDCG@k (Discounted Cumulative Gain)

## Problem

Ranking tasks (RAG retrieval, search, rerankers, recommendation) cannot be scored with token accuracy: putting the wrong document at rank 1 costs far more than at rank 10, and relevance is **graded** (0 = useless, 3 = perfect answer). nDCG (Järvelin & Kekäläinen, 2002) bakes both facts into one formula: each position contributes a **gain** determined by the label times a **discount** that decays with rank, the sum is the DCG, and dividing by the IDCG of the ideal ordering normalises the result to \([0, 1]\) so different queries and different candidate-pool sizes become comparable.

It is the de-facto standard for RAG evaluation and the headline metric of the BEIR / MTEB leaderboards. Contrast it with RRF (`retrieval-rrf-fusion`): RRF is a **fusion** algorithm that consumes only ranks, ignores graded labels and emits a new ordering; nDCG is an **evaluation** metric that consumes graded labels and emits a score.

## Rules

- There are \(N\) candidate documents; a document's id is its **index** in the lists below.
- `scores`: `list[float]` — the ranking model's scores (any sign, used only for comparisons).
- `rels`: `list[int]` — graded relevance labels aligned one-to-one with `scores`; non-negative integers, larger is better.
- `k`: the cut-off. The number of positions actually scored is \(m = \min(k, N)\); \(k > N\) behaves exactly like \(k = N\).
- Ordering: sort by `scores` **descending**. Documents with equal scores are ordered by **ascending index** (a stable tie-break), so the outcome depends on the index only, never on how the list was built.
- Gain, chosen by the explicit `exp_gain` switch (exponential by default):
  \[
  g(r) = 2^{r} - 1 \quad (\text{exponential gain, amplifies top labels})\qquad
  g(r) = r \quad (\text{linear gain})
  \]
- Discount: rank \(i\) starts at **1** and the factor is \(1 / \log_2(i + 1)\). Hence position 1 has discount \(1/\log_2 2 = 1\) (no discount at all), position 2 has \(1/\log_2 3 \approx 0.6309\).
- Accumulation:
  \[
  \text{DCG} @ m = \sum_{i=1}^{m} \frac{g(r_{\pi(i)})}{\log_2(i+1)},
  \qquad
  \text{IDCG} @ m = \sum_{i=1}^{m} \frac{g(r_{\sigma(i)})}{\log_2(i+1)}
  \]
  where \(\pi\) is the ordering produced by the model and \(\sigma\) is the labels sorted **descending** (the ideal ordering) — IDCG depends only on the labels, never on the scores.
- Normalisation and sentinel:
  \[
  \text{nDCG} @ k = \begin{cases} \text{DCG} / \text{IDCG}, & \text{IDCG} > 0 \\ 0.0, & \text{IDCG} = 0 \end{cases}
  \]
  \(\text{IDCG} = 0\) happens when "every label is 0" or "there are no candidates / \(k \le 0\)"; then nDCG is defined as **0.0** (not NaN, not 1.0) and DCG is necessarily \(0\) too.
- Return the triple `(ndcg, dcg, idcg)` as Python floats. IDCG uses the **same** \(m\) and the same gain/discount formulas, so \(0 \le \text{nDCG} \le 1\) always. Using `sklearn` is forbidden (it is not installed either).

## Example

```
scores = [0.10, 0.85, 0.40, 0.85, 0.25, 0.60]
rels   = [  0,    3,    2,    1,    2,    3]
k = 4, exp_gain = True

doc index:    0     1     2     3     4     5
score:      0.10  0.85  0.40  0.85  0.25  0.60
rank:         6     1     4     2     5     3
0.85 occurs twice (docs 1 and 3) => ascending index puts doc 1 at rank 1, doc 3 at rank 2.
model ordering = [1, 3, 5, 2 | 4, 0], labels of the first 4 positions = [3, 1, 3, 2]

rank i   :   1         2         3       4
label    :   3         1         3       2
gain 2^r-1:  7         1         7       3
discount   : 1/log2 2  1/log2 3  1/log2 4  1/log2 5  = 1, 0.630930, 0.5, 0.430677
terms      : 7.0      0.630930  3.5      1.292030
DCG@4 = 12.422959

ideal labels = [3, 3, 2, 2, 1, 0], first 4 gains = [7, 7, 3, 3]
terms        : 7.0      4.416512  1.5      1.292030
IDCG@4 = 14.208538

nDCG@4 = 12.422959 / 14.208538 = 0.874331
```

The same data with linear gain (`exp_gain = False`):

```
gain = r:  DCG  = 3 + 1(0.630930) + 3(0.5) + 2(0.430677) = 5.992283
           IDCG = 3 + 3(0.630930) + 2(0.5) + 2(0.430677) = 6.754142
           nDCG = 0.887201
```

Ties are expensive: if an implementation puts doc 3 (label 1) ahead of doc 1 (label 3), DCG falls to \(10.208538\) and nDCG becomes \(0.718479\) — a single wrong tie-break costs **15.6 points**.

## Constraints

- CPU; `scores` and `rels` have the same length; \(N \le 10^4\) (\(N = 0\) allowed).
- \(0 \le \text{nDCG} \le 1\); nDCG is **not monotone** in \(k\) (shorter cut-offs shrink IDCG as well), while DCG and IDCG are monotone non-decreasing in \(m\).
- **Degenerate behaviour**: model ordering equals the ideal ordering ⇒ nDCG = 1; all labels 0 ⇒ `(0.0, 0.0, 0.0)`; `k >= N` ⇒ the whole list is scored; `k = 1` with a top-labelled document first ⇒ nDCG = 1; any **strictly increasing** transform of the scores (add a constant, multiply by a positive number, push through a sigmoid) leaves the ordering — and therefore the score — untouched: nDCG reads the order, not the values.
- Versus `retrieval-rrf-fusion`: that problem takes **rank lists from several systems**, has no graded labels and must **produce a fused ranking**; here the input is **one ranking plus human labels** and the output is a **score**.
