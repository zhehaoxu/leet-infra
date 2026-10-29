# Retrieval: Full BM25 Scoring

## Problem

BM25 is the baseline scoring function of sparse retrieval (Robertson & Walker, SIGIR 1994; Lucene `BM25Similarity`): it combines "a term appearing more often means more relevant" (term-frequency saturation) with "longer documents are penalised" (length normalisation) into an additive score, and uses the inverse document frequency \( \mathrm{idf}(t) \) to down-weight terms that occur everywhere. In real systems it is the keyword path of hybrid search and an upstream of fusers such as RRF.

This problem starts from a **pure token-id corpus**: compute the document frequency \( \mathrm{df}(t) \) and the average document length \( \mathrm{avgdl} \) on the fly, accumulate a full BM25 score for every query token, and return the dense corpus scores plus the top-k indices. Three things bite:

- two idf conventions: Lucene's \( \log(1+\cdot) \) is always positive, while Robertson's original form becomes **negative** when \( \mathrm{df}(t)>N/2 \), and it must be kept as-is rather than clamped to 0;
- a token repeated in the query is accumulated once per **occurrence** (no de-duplication), which follows from BM25's additivity over query terms;
- score ties break by ascending document index, and empty queries, out-of-corpus tokens and \( \mathrm{avgdl}=0 \) all have pinned degenerate answers.

CPU, fully deterministic, no RNG calls.

## Rules

The full signature, parameter meanings and defaults live only in the starter docstring; that is the sole judging contract.

- The corpus is an \((N,L)\) int64 tensor. The first `doc_lens[d]` entries of row \(d\) are document \(d\)'s token-id sequence; the remaining entries are padding (any content, `-1` allowed) and are **always ignored**. \(L\) may be 0, but \(N\ge 1\).
- The document frequency \( \mathrm{df}(t) \) is "the number of documents whose valid prefix contains token \(t\)", computed from the corpus; \( \mathrm{avgdl}=\frac{1}{N}\sum_d |d| \) with \( |d|=\text{doc\_lens}[d] \).
- `idf_mode` selects one of two formulas: `"lucene"` uses \( \mathrm{idf}(t)=\log\!\left(1+\frac{N-\mathrm{df}(t)+0.5}{\mathrm{df}(t)+0.5}\right) \); `"robertson"` uses \( \mathrm{idf}(t)=\log\!\left(\frac{N-\mathrm{df}(t)+0.5}{\mathrm{df}(t)+0.5}\right) \). The latter is negative when \( \mathrm{df}(t)>N/2 \); negative scores are legal and \( \max(0,\mathrm{idf}) \) is forbidden.
- Document \(d\)'s contribution for one occurrence of token \(t\) in the query is
  \( \mathrm{idf}(t)\cdot\frac{\mathrm{tf}(t,d)\,(k_1+1)}{\mathrm{tf}(t,d)+k_1\left(1-b+b\,|d|/\mathrm{avgdl}\right)} \),
  where \( \mathrm{tf}(t,d) \) counts \(t\) inside document \(d\)'s valid prefix. The document score is the sum over every query occurrence; a token appearing twice in the query contributes twice.
- When \( \mathrm{avgdl}=0 \) (all documents empty) the length ratio is taken as 0 to avoid \(0/0\); every score is then 0.
- A query token absent from the corpus has \( \mathrm{df}=0 \) and all \( \mathrm{tf}=0 \), so it contributes 0, while its idf is still defined by the formula.
- Return a pair: `scores` is \((N,)\) **float64**; `topk` is \(( \min(k,N), )\) **int64**, sorted by (score descending, document index ascending). `k=0` yields an empty tensor and `k>N` returns all \(N\).
- An empty query gives all-zero scores and `topk` equal to \( [0,1,\dots,\min(k,N)-1] \).
- Illegal input raises `ValueError`: malformed shape/dtype, out-of-range `doc_lens`, negative valid/query token ids, \(N=0\), `k<0`, `k1<=0`, `b` outside \([0,1]\), unknown `idf_mode`.

## Example

Corpus \(N=3\), \(L=4\), `doc_lens = [4,3,2]`, query `[2,2,4]` (token 2 occurs twice), \(k_1=1.2\), \(b=0.75\):

```
d0: [1, 2, 2, 3]   |d0|=4  tf(2)=2
d1: [2, 3, 4, 0]   |d1|=3  tf(2)=1, tf(4)=1
d2: [1, 4, -1, -1] |d2|=2  tf(4)=1      (-1 is padding and is ignored)
avgdl = (4+3+2)/3 = 3
df(2) = 2 (d0,d1), df(4) = 2 (d1,d2)
```

lucene: \( \mathrm{idf}(2)=\mathrm{idf}(4)=\log(1+1.5/2.5)=\log 1.6\approx 0.4700036292 \).

```
score(d0) = 2 × [0.4700036 × 2×2.2/(2+1.5)]                 = 1.1817234107
score(d1) = 2 × [0.4700036 × 1×2.2/(1+1.2)] + 0.4700036×1   = 1.4100108877
score(d2) = 0.4700036 × 1×2.2/(1+1.2×(0.25+0.75×2/3))       = 0.5442147286
topk (k=3) = [1, 0, 2]
```

robertson: \( \mathrm{df}=2>N/2=1.5 \), so \( \mathrm{idf}=\log(0.6)\approx-0.5108256238<0 \); scores turn negative and the order reverses:

```
scores = [-1.2843615683, -1.5324768713, -0.5914823012]
topk   = [2, 0, 1]
```

De-duplicating the query to `[2,4]` would instead give \( [0.5908617,\ 0.9400073,\ 0.5442147] \). The top-1 is still d1 here, but the numbers differ completely — enough for a de-duplicating implementation to fail.

## Constraints

- Return `(scores, topk)`: `scores` is \((N,)\) float64 (zeros and negatives included), `topk` is int64 with length \( \min(k,N) \).
- Judging uses absolute tolerance \(10^{-9}\) for `scores` (float64) and `torch.equal` for `topk`; the implementation path is free but accumulation must stay in float64.
- Ties strictly break by ascending document index; empty query, OOV token, \( \mathrm{avgdl}=0 \), `k=0` and `k>N` must produce the degenerate answers above instead of raising.
- CPU, deterministic, no random numbers; the `doc_lens`/padding convention is part of the contract — padding must never enter `df` or `tf`.
