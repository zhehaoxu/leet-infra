# Deep dive: nDCG@k

## Origin
- Kalervo Järvelin & Jaana Kekäläinen, *Cumulated Gain-based Evaluation of IR Techniques*, ACM TOIS 20(4), 2002. Their motivation: precision/recall only ask "how many of the top \(k\) are hits", treating a hit at rank 1 and a hit at rank 10 as equivalent, and they cannot exploit graded relevance (0 = useless … 3 = perfect). Hence
  \[ \text{CG}_k = \sum_{i=1}^{k} r_i,\qquad
     \text{DCG}_k = \sum_{i=1}^{k} \frac{r_i}{\log_2(i+1)} \]
  the \(1/\log_2(i+1)\) discount was picked by **human satisfaction experiments**: it matches how people perceive rank better than a linear or a \(1/i\) discount.
- The normaliser IDCG comes from the same paper: the DCG of the ideal ordering of the very same labels, which is what makes \(0 \le \text{nDCG} \le 1\) comparable across queries and across candidate-pool sizes. The paper offers both \(r_i\) as-is (**linear / arithmetic gain**) and \(2^{r_i}-1\) (**exponential gain**); the latter amplifies high labels and is the default in modern tooling.
- Modern implementations: `sklearn.metrics.ndcg_score` (`norm='exponential'|'arithmic'`, discount \(\log_2(\text{rank}+1)\) with rank starting at 1), `torchmetrics.RetrievalNormalizedDCG` (`gain_function` selectable), RankLib, pytrec_eval (the BEIR/MTEB leaderboards run on it, with a gain convention that differs from sklearn's default). When two "nDCG" numbers disagree, the first two things to check are always the **gain form** and the **rank origin**; sklearn is not installed in the judge environment either, so the metric must be written from the definition.

## Full derivation and index conventions
Let the candidates be \(d_0,\dots,d_{N-1}\) with model scores \(s_i\) and human labels \(r_i \in \mathbb{Z}_{\ge 0}\).

1. **Ordering.** Take a permutation \(\pi\) sorting \(s\) descending, with "equal scores ⇒ ascending index". This tie-break is not optional: it is the only convention that makes an implementation deterministic (a stable sort, or sorting by the key \((-s_i, i)\), are equivalent).
2. **Cut-off.** \(m = \min(k, N)\). DCG sums only the first \(m\) ranks; \(k > N\) is exactly \(k = N\) — there is simply nothing left to place, the sequence is not zero-padded.
3. **Gain.**
  \[ g(r) = \begin{cases} 2^{r}-1, & \text{exponential (default)}\\ r, & \text{linear}\end{cases} \]
  \(g(0) = 0\) under both, so an irrelevant document at the front contributes nothing to the numerator **while pushing out** a position that could have — that is precisely the penalty.
4. **Discount.** Rank \(i\) starts at 1 with \(c_i = 1/\log_2(i+1)\): \(c_1 = 1\) (the top slot is never discounted), \(c_2 \approx 0.6309\), \(c_3 = 0.5\), \(c_5 \approx 0.4307\). Writing \(1/\log_2 i\) divides by zero at \(i=1\) (\(\infty\)/NaN); writing \(1/\log_2(i+2)\) (the classic 0-based slip) discounts every position once more, so DCG is systematically low — **but if IDCG is wrong in the same way the ratio partially cancels**, which is why the problem asks for `dcg` and `idcg` to be returned as well.
5. **Accumulation and the ideal value.**
  \[ \text{DCG}@m = \sum_{i=1}^{m} g(r_{\pi(i)})\,c_i,\qquad
     \text{IDCG}@m = \sum_{i=1}^{m} g(r^{\downarrow}_i)\,c_i,\quad r^{\downarrow} = \text{sort}(\text{rels}, \text{desc}) \]
  IDCG depends only on the multiset of labels, never on the scores — a frequent misconception is that the ideal ranking must be produced by "some perfect model".
6. **Normalisation and sentinel.**
  \[ \text{nDCG}@k = \begin{cases} \text{DCG}/\text{IDCG}, & \text{IDCG} > 0\\ 0.0, & \text{IDCG} = 0\end{cases} \]
  \(\text{IDCG} = 0 \iff\) all gains of the top \(m\) ideal labels are \(0 \iff\) every \(r_i = 0\) (or \(m = 0\)). Ranking quality is then undefined; the standard is to report 0, not NaN, and reporting 1 would hand a perfect score to a system retrieving from a corpus with nothing relevant.

## Core traps
1. **Discount off-by-one** (item 4 above) in its two usual shapes: a 0-based \(i\) fed into \(1/\log_2(i+1)\) (top rank becomes \(\log_2 1 = 0\), i.e. division by zero), or numerator and denominator using different \(m\).
2. **Forgetting to truncate IDCG to the same \(m\)**: dividing \(\text{DCG}@k\) by the IDCG of the full list depresses every short cut-off and makes "why is my nDCG@10 three points below theirs" unanswerable.
3. **Unstable tie-break**: `np.argsort` defaults to quicksort and does not guarantee stability; torch's `argsort` needs an explicit `stable=True` (or a secondary key on the original index). Ties are common in real systems (one embedding matching several chunks, rerankers whose float16 outputs collapse onto the same value); without a fixed tie-break, evaluating the same checkpoint twice yields two different nDCG values — a textbook case of irreproducible evaluation. The tied example in this problem (two 0.85 scores) exists exactly to catch it: getting it wrong moves \(0.874331 \to 0.718479\).
4. **Sorting `scores` but not carrying `rels` along**: gather the labels with the argsort **indices**; sorting the two lists independently, or using `sorted(rels)` as if it were the model order, fails silently.
5. **\(2^{r}-1\) overflow and dtype**: labels around 60 already exceed float32's exact-integer range; the reference solution keeps float64 intermediates and only then returns Python floats. Also \(r\) must be an integer — if upstream hands you floats, the rounding convention has to match the labelling guide.
6. **Division by zero / NaN propagation**: skipping the sentinel branch when IDCG is 0 gives NaN, and one `mean` over the benchmark turns everything into NaN. For batches, return per-query nDCG (including the 0s) and average over queries, i.e. the **macro** view — same micro/macro split as in `eval-token-accuracy`.
7. **Do not index past \(N\) when \(k > N\)**: compute `min(k, N)` before indexing.

## Versus neighbouring metrics / algorithms
- **vs `retrieval-rrf-fusion`**: RRF consumes **rank lists from several systems**, weights them like \(1/(k+i)\), uses no human labels at all, and outputs **a ranking**; nDCG consumes **one ranking plus graded labels** and outputs **a score**. RRF has no notion of gain; nDCG has no notion of fusion.
- **vs `retrieval-mmr`**: MMR is a **query-time re-ranking policy** (relevance traded against diversity); nDCG is an **after-the-fact score**. MMR's diversity benefit only becomes quantifiable once graded labels and nDCG are applied to it.
- **vs precision@k / recall@k / hit-rate**: those use binary labels and treat all positions inside the cut-off alike; substituting 0/1 labels and linear gain into nDCG degenerates it into a discount-weighted precision-like measure, which is still not P@k.
- **vs `eval-bleu-score` / `eval-rouge-score` / `eval-token-accuracy`**: that family scores the **generated content itself** (n-grams, LCS, position-by-position alignment); nDCG scores the **order of a candidate set** — different inputs, different references, different lengths.
- **vs `eval-shingle-contamination`**: contamination is data auditing (did the eval set leak into training), nDCG is ranking quality; both appear in the same RAG report but share no formula.

## Cross-check design
Reference solution: `torch.tensor(float64)` + `torch.argsort(descending=True, stable=True)` for the order, one `torch.sort` for the ideal sequence, vectorised `torch.pow/log2/arange` accumulation. The judge's `_oracle` takes a completely different road: a **hand-written selection sort** that scans the remaining documents each round and swaps only on a **strictly greater** score (which is what naturally yields the ascending-index tie-break), gains built by **repeated doubling** instead of `2**r`, discounts via `math.log(i+2, 2)` in a 0-based loop, the ideal sequence via **insertion sort**, and pure Python scalar accumulation with no tensor involved.
Beyond the example the suite asserts: ideal ordering ⇒ 1.0 and DCG == IDCG; \(k > N\) identical to the full list with DCG/IDCG monotone non-decreasing in \(m\); all-zero labels ⇒ `(0.0, 0.0, 0.0)`; empty input and \(k = 0\); the tied-score DCG recomputed term by term; single document; permutation invariance when scores are pairwise distinct; exponential gain below linear gain with exact values; plus 8 randomised batches (negative scores, \(k > N\), both gains) cross-checked throughout with `isclose(atol=1e-5, rel_tol=1e-5)`.

## Complexity
Sorting \(O(N \log N)\) with a stable sort; if only the first \(k\) ranks are needed, selection or a heap gives \(O(Nk)\) or \(O(N + k\log N)\). DCG/IDCG cost \(O(k)\) multiply-adds each, space is \(O(N)\) (labels and rank arrays). A batch evaluation is independent per query, total cost \(\sum_q O(N_q \log N_q)\).

## Practical notes
nDCG@10 and nDCG@5 routinely differ by 1–2 points, so any reported number must name \(k\). Changing the label scale (0–3 vs 0–4 vs 0–10) rescales \(2^r-1\) exponentially, which makes cross-dataset comparisons meaningless. When labels come from LLM-as-judge, freeze the prompt and the rubric: otherwise the IDCG denominator itself drifts between runs.
