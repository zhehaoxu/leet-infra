# Retrieval: Postings Lists and Block-Max WAND

## Problem

An inverted index flips "document → term" into "term → document" and is the storage substrate of BM25 sparse retrieval. Query evaluation has two complementary halves:

- **Boolean AND**: find the documents containing every query term. Each postings list is docid-ascending, so the last docid of a block can serve as a **skip pointer**, letting the intersection leapfrog over long runs instead of scanning row by row;
- **Ranked top-k**: WAND-family algorithms compare "the sum of per-term score upper bounds for the current candidate" with the current top-k threshold and prune whole blocks of documents that cannot enter the list. Block-Max WAND (Ding & Suel, SIGIR 2011) tightens the bound from a whole-list maximum to a per-block maximum, pruning more aggressively — but the bound must remain a true upper bound: pruning must never change the final top-k.

This problem chains the two. The input is a set of already-built postings (each a docid-ascending list of `(docid, tf)` rows), cut into blocks of `block_size` (the last docid of each block is the skip target). Return the skip-AND candidates, the Block-Max WAND top-k, and the number of **whole blocks pruned**. The score is BM25 with the Lucene idf, with formula and parameters pinned in the starter docstring.

## Rules

The full signature, parameter meanings and defaults live only in the starter docstring; that is the sole judging contract.

- `postings` is a list/tuple whose length equals the number of query-term occurrences (an empty list is an empty query). Each element is an \((n_i,2)\) int64 postings list of `(docid, tf)` rows; `docid` must be **strictly increasing** and inside \([0,N)\), and `tf>=1`. Shape \((0,2)\) means the term is absent from the corpus; the same term occurring several times is passed as several copies of the list and contributes once per occurrence.
- The corpus size \(N\) comes from `doc_lens` (an int64 vector of length \(N\ge1\)); \( |d|=\text{doc\_lens}[d] \) and \( \mathrm{avgdl}=\frac1N\sum_d |d| \). When \( \mathrm{avgdl}=0 \) the length ratio is taken as 0.
- Scoring is BM25 with Lucene idf: for list \(i\), \( \mathrm{df}_i=n_i \) and \( \mathrm{idf}_i=\log\!\left(1+\frac{N-\mathrm{df}_i+0.5}{\mathrm{df}_i+0.5}\right) \); document \(d\)'s score is the sum over every list containing \(d\) of
  \( \mathrm{idf}_i\cdot\frac{\mathrm{tf}_{i,d}(k_1+1)}{\mathrm{tf}_{i,d}+k_1(1-b+b|d|/\mathrm{avgdl})} \).
  Documents absent from every postings list score 0 but are not top-k candidates.
- **skip-AND**: return the docids present in **all non-empty** postings lists, ascending. An empty query, or any empty list, yields an empty intersection. Blocks are consecutive runs of `block_size` rows (the last may be short), and a block's last docid is its skip target; pointers only move forward in docid.
- **Block-Max WAND**: the upper bound of a block is the maximum BM25 contribution of its rows. Compare the sum of bounds against the current top-k threshold (the \(k\)-th best score, or 0 while fewer than \(k\) documents are held) and skip whole blocks that cannot beat it; each such block increments `pruned_blocks`. A block that was partially scanned does not count. The pruned top-k must be identical to a brute-force full-scoring ranking.
- The top-k domain is the **union** of documents appearing in at least one postings list, of size \(U\); `topk` has length \( \min(k,U) \), sorted by (score descending, docid ascending). `k=0` is empty and `k>U` returns all \(U\).
- Return a triple: `candidates` is \((C,)\) int64, `topk` is \(( \min(k,U), )\) int64, and `pruned_blocks` is a non-negative Python `int`. Empty query or `k=0` implies `pruned_blocks=0`.
- Illegal input raises `ValueError`: `postings` not a list/tuple, an element not an \((n,2)\) integer tensor, non-increasing or out-of-range docid, `tf<1`, malformed `doc_lens`, \(N=0\), `k<0`, `k1<=0`, `b` outside \([0,1]\), `block_size<1`.

## Example

\(N=6\), `doc_lens = [4,4,4,4,4,4]` (so \( \mathrm{avgdl}=4 \)), `block_size=2`, three postings lists:

```
term0: (0,2) (1,1) (2,1) (3,1) (4,1) (5,2)   blocks: [0,1] [2,3] [4,5]
term1: (0,1) (5,3)                            blocks: [0,5]
term2: (0,2) (5,2)                            blocks: [0,5]
```

\( \mathrm{df}_0=6 \), \( \mathrm{idf}_0=\log(1+0.5/6.5)\approx0.0741080 \); \( \mathrm{df}_1=\mathrm{df}_2=2 \), \( \mathrm{idf}_1=\mathrm{idf}_2=\log(1+4.5/2.5)=\log2.8\approx1.0296194 \).

```
score(d0) = 0.1018985 + 1.0296194 + 1.4157267 = 2.5472445775
score(d1..d4) = 0.0741080                      (term0 only, tf=1)
score(d5) = 0.1018985 + 1.6179734 + 1.4157267 = 3.1355985302
```

- `candidates = [0, 5]` (all three lists contain 0 and 5).
- With `k=2`: `topk = [5, 0]` and `pruned_blocks = 1`. After doc0 (2.5472) and doc1 (0.0741080) are scored, the threshold is 0.0741080; term0's block `[2,3]` has block maximum exactly 0.0741080, and only a **strictly greater** score can enter, so the whole block is skipped. doc2 and doc3 would tie with doc1 but have larger docids, so they lose the tie anyway — skipping is safe.
- With `k=3`: `pruned_blocks = 0`, because doc2 is scored before the heap fills, so block `[2,3]` is only partially scanned and does not count as a whole-block prune.

## Constraints

- `candidates` and `topk` are 1-D int64 tensors and `pruned_blocks` is a Python `int`; judging compares the tensors with `torch.equal` and the count with integer equality.
- Correct pruning is a hard requirement: topk must equal brute-force full scoring over the whole union; bounds may be loose but never tight.
- Docid monotonicity, skip jumps, block bounds, threshold ties (strict `>`) and the "whole block" convention for `pruned_blocks` are all part of the contract.
- CPU, deterministic, no random numbers; `block_size=1` and a short final block are both legal.
