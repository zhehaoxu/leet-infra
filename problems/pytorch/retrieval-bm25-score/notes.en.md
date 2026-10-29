# In Depth: Retrieval: Full BM25 Scoring

## Source

- Original BM25: Robertson & Walker, *Some simple effective approximations to the 2-Poisson model for probabilistic weighted retrieval*, SIGIR 1994 (the TREC-3 paper). It gives the probabilistic-retrieval form of term-frequency saturation plus length normalisation used here: \( \mathrm{tf}(k_1+1)/(\mathrm{tf}+k_1(1-b+b|d|/\mathrm{avgdl})) \).
- Engineering conventions for the two idfs: the Lucene `BM25Similarity` official documentation (Javadoc) explicitly uses \( \log(1+(N-\mathrm{df}+0.5)/(\mathrm{df}+0.5)) \); the Robertson form \( \log((N-\mathrm{df}+0.5)/(\mathrm{df}+0.5)) \) is negative when \( \mathrm{df}>N/2 \), and Lucene's `+1` exists precisely to avoid that. This problem implements both, but the robertson mode keeps the original form **without clamping**.
- Application context: the default similarity in Elasticsearch/Lucene, the sparse-retrieval baseline, and the keyword path of hybrid search (fusion is `retrieval-rrf-fusion`). This task only does "full scoring of a given corpus and query"; it does not build or compress an index.
- CPU, fully deterministic: df/tf are counts and idf is `math.log`; no RNG.

## Full derivation and index convention

**Corpus and valid prefix**: `docs` is \((N,L)\), and row \(d\)'s valid range is its first \( |d|=\text{doc\_lens}[d] \) entries. Padding values never enter any statistic and may be anything (`-1` in the example). Hence

\[
\mathrm{df}(t)=\sum_{d=0}^{N-1}\mathbf{1}\!\left[t\in \text{docs}[d,:|d|]\right],
\qquad
\mathrm{avgdl}=\frac{1}{N}\sum_{d=0}^{N-1}|d|.
\]

The reference builds a broadcast mask `positions = arange(L) < doc_lens[:,None]`, then for each query token computes `(positions & (docs == t)).sum(dim=1)` for tf and `(tf > 0).sum()` for df. It builds no vocabulary and calls no `bincount`, so token ids need not lie in a fixed vocab.

**The BM25 contribution**: for one query occurrence of token \(t\),

\[
c_t(d)=\mathrm{idf}(t)\cdot
\frac{\mathrm{tf}(t,d)(k_1+1)}
     {\mathrm{tf}(t,d)+k_1\left(1-b+b\,|d|/\mathrm{avgdl}\right)}.
\]

The denominator term \(k_1(1-b+b|d|/\mathrm{avgdl})\) is the length penalty: \(b=1\) scales fully with relative length and \(b=0\) degenerates to pure saturation. \(k_1\) controls saturation speed; as \(k_1\to0\), \(c_t(d)\to \mathrm{idf}(t)\cdot \mathbf{1}[\mathrm{tf}>0]\) (presence only).

**Two idfs and negative values**: lucene's \( \log(1+x) \) is positive for every \(x>0\); robertson's \( \log x \) is negative exactly when

\[
\frac{N-\mathrm{df}+0.5}{\mathrm{df}+0.5}<1
\iff \mathrm{df}>\frac N2.
\]

For the example \(N=3,\mathrm{df}(2)=\mathrm{df}(4)=2>1.5\), measured

\[
\mathrm{idf}_{\text{lucene}}=\log 1.6=0.47000362924573563,\qquad
\mathrm{idf}_{\text{robertson}}=\log 0.6=-0.5108256237659907,
\]

with ratio \( -1.0868546368158187 \). So under robertson all three scores are the same negative multiple of the lucene scores:

```
lucene     : [1.1817234107, 1.4100108877, 0.5442147286]      -> topk [1,0,2]
robertson  : [-1.2843615683, -1.5324768713, -0.5914823012]   -> topk [2,0,1]
```

The ranking reverses. An implementation that writes `max(0, idf)` turns robertson into all-zero scores and `topk` into `[0,1,2]` instead of `[2,0,1]` — the sharpest discriminator in this problem.

**Linearity in repeated query tokens**: BM25 is additive over query terms, so a token occurring \(m\) times in the query multiplies the single-occurrence score by \(m\):

\[
\mathrm{score}(q=[t,t])=\sum_d 2c_t(d)=2\,\mathrm{score}(q=[t]).
\]

Measured on the example, `score([2,2])` equals `2*score([2])` element-wise (within `1e-12`) and `score([2,2,4]) = 2*score([2]) + score([4])`. The decomposition of the three documents is

```
score([2]) = [0.5908617, 0.4700036, 0.0]
score([4]) = [0.0,       0.4700036, 0.5442147]
score([2,2,4]) = [1.1817234, 1.4100109, 0.5442147]
```

d1 beats d0 because it collects two copies of token 2 and one of token 4, while d0 only collects two copies of token 2. De-duplicating the query to `[2,4]` gives `[0.5908617, 0.9400073, 0.5442147]`; the top-1 is still d1, but every number changes.

**The avgdl=0 degenerate case**: when all documents are empty, \( |d|/\mathrm{avgdl} \) is \(0/0\). The contract takes the ratio as 0, so the denominator is \( \mathrm{tf}+k_1(1-b) \); every tf is already 0, hence every score is 0. The reference uses `torch.where(tf > 0, ..., 0)` to mask out the `tf=0` divisions, and measurement confirms no NaN. Note that with \(b=1\) and \( |d|=0 \) a single document's norm can also be 0, but its tf is 0 and the same mask covers it.

**Empty query and OOV**: an empty query has no terms, so `scores` is all zeros and `topk` is the first \( \min(k,N) \) indices after a stable sort, i.e. `[0,1,...]`. An OOV token has \( \mathrm{df}=0 \), giving lucene idf \( \log(1+(N+0.5)/0.5) \) and robertson idf \( \log((N+0.5)/0.5) \), but all tf are 0 so the contribution is 0. **Do not raise or skip idf for df=0**; that would conflate "OOV scores 0" with "OOV is illegal".

**Top-k tie convention**: the contract is (score descending, index ascending). The reference uses `torch.argsort(-scores, stable=True)`: a stable sort keeps equal elements in their original index order, which is exactly ascending index; this matches Python's `sorted(range(N), key=lambda d: (-score[d], d))`. The `k` cut happens after sorting, so `k>N` naturally returns all \(N\).

**dtype**: scoring stays in float64 throughout (`doc_lens` cast to float64, tf cast to float64, idf a Python float); scores are float64 and topk int64. Accumulating in float32 fails the \(10^{-9}\) tolerance on the multi-token, negative-idf cases.

## Key points and common mistakes

1. **df/avgdl must be computed from the valid prefix only**: counting padding in df/tf turns d2's `-1` into a real token and breaks both scores and topk.
2. **Never clamp the robertson negative idf**: for `df > N/2` the negative value is part of the definition; `max(0, idf)`, `abs(idf)` and clamping scores to 0 are all wrong. The example's `[2,0,1]` exists to catch this.
3. **Repeated query tokens accumulate per occurrence**: de-duplication is a different convention, and judging probes `score([2,2]) == 2*score([2])` directly.
4. **avgdl=0 is a 0/0 case**: take the length ratio as 0 and mask tf to avoid NaN; both branches of `torch.where` are evaluated, so a zero denominator must not leak NaN into the output.
5. **Ties break by ascending document index**: `argsort(descending=True)` is not stable by default; use `stable=True` or `argsort(-scores, stable=True)`. The all-equal three-document case catches this.
6. **Three k degeneracies**: `k=0` returns empty (not an error), `k>N` returns all \(N\) (not zero-padded), and an empty query gives the index prefix (not an empty result).
7. **OOV idf is still defined while its contribution is 0**: do not treat df=0 as illegal input.
8. **Valid and query tokens are non-negative**: padding may be -1, but a -1 inside the valid prefix must raise `ValueError`; the `doc_lens` mask is what distinguishes the two.
9. **Floating-point precision**: float64 is mandatory; float32 is not enough for the \(10^{-9}\) absolute tolerance.
10. **The b boundaries**: both \(b=0\) and \(b=1\) are legal; with \(b=1\) and an empty document norm=0, which the tf mask must cover.
11. **N=0 is illegal**: avgdl is undefined, so the contract raises `ValueError` instead of returning empty tensors.

## Differences from neighboring problems

- **vs `retrieval-rrf-fusion` (same collection)**: RRF uses **ranks only**, fusing several result lists by \(1/(k+\text{rank})\) and ignoring term statistics, tf and idf. This problem is the sparse-scoring **body itself**; its scores and ranks are what RRF consumes. RRF's input is an ordered list, this problem's input is a raw token corpus.
- **vs `retrieval-rag-chunk` (same collection)**: RAG chunking outputs only an integer matrix of "which token indices go into which chunk" and performs no scoring; this problem consumes token ids and produces real-valued scores — the data roles are reversed.
- **vs `retrieval-ivf-assign` (same collection)**: IVF is a dense Euclidean argmin over vectors (coarse-quantiser assignment) with no terms, df or inverted lists; this problem is sparse term statistics plus BM25.
- **vs `retrieval-postings-wand` (same batch)**: that problem takes already-built postings, returns only a union top-k pruned by block bounds, and never materialises dense scores; this problem takes a dense token corpus and returns dense corpus scores, with df recomputed from the corpus. They share the BM25 formula but differ in data layout, output and algorithmic goal.
- **vs `searchsorted-bincount`**: that is a "value → bucket + count" histogram primitive with no retrieval scoring; no boundaries or buckets exist here.

## Cross-check convention and complexity

- **Primary oracle**: `_oracle_bm25` is pure-Python nested loops — first scan each document's valid tokens for df, then use `list.count` per document for tf and accumulate per query occurrence. It is structurally different from the reference's "broadcast comparison + tensor sums". It covers five corpora: repeated query terms, a zero-length document, robertson with `k1=0.9/b=0.3`, `k1=2.0/b=0.0`, and a single document; scores use `atol=1e-9, rtol=0`, topk uses `torch.equal`.
- **Discriminating cases**: the lucene/robertson example (constants filled from measurements), `score([2,2]) == 2*score([2])` and `score([2,2,4]) == score([2,2]) + score([4])`, robertson scores all negative and a fixed negative multiple of lucene, empty query, OOV under both idfs, three fully tied documents with ascending indices, all-empty documents (no NaN), `k=0/1/N/100`, and 13 invalid-input groups.
- **Non-obvious measured facts**: in the example d1's 1.4100108877 beats d0's 1.1817234107 because the repeated token 2 contributes twice in d1; the robertson negative idf flips the ranking from `[1,0,2]` to `[2,0,1]`; all-empty documents under `avgdl=0` must yield 0, not NaN.
- **Complexity**: the reference performs one \(O(NL)\) masked comparison and sum per query occurrence, so \(O(QNL)\) time and an \(O(NL)\) temporary boolean matrix, plus an \(O(N\log N)\) sort. At judging sizes (\(N\le 6\), \(L\le 8\), \(Q\le 4\)) this is far inside the limit. For a large corpus one would build a token-to-postings map and use `index_add_`, which this task does not require.
