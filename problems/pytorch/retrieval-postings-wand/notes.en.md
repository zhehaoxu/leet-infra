# In Depth: Retrieval: Postings Lists and Block-Max WAND

## Source

- Threshold upper-bound idea of WAND: Broder, Carmel, Herscovici, Soffer, Zien, *Efficient query evaluation using a two-level retrieval process*, CIKM 2003; the block-level variant: Ding & Suel, *Faster top-k document retrieval using block-max indexes*, SIGIR 2011. The pivot / block-bound / whole-block-skip scheme here belongs to that family; the exact pseudocode follows this task's docstring (pivot details differ slightly between papers and were not copied line by line).
- Skip pointers: Manning, Raghavan, Schütze, *Introduction to Information Retrieval*, 2008, chapter 2 (skip pointers in boolean retrieval). This task uses "the last docid of each block" as the skip target, with block size derived from `block_size`.
- Scoring: BM25 with the Lucene `BM25Similarity` idf (official Javadoc), the same formula as `retrieval-bm25-score`; here df is simply the postings length.
- CPU, fully deterministic: every advance is driven by docid comparisons and fixed block boundaries; no RNG.

## Full derivation and index convention

**Postings and blocks**: list \(i\) is ascending \( (d_{i,0},f_{i,0}),\dots,(d_{i,n_i-1},f_{i,n_i-1}) \) with strictly increasing docids. With `block_size = B` it is cut into row ranges

\[
[bB,\ \min((b+1)B,n_i)-1],\qquad b=0,1,\dots
\]

and the last block may be short. The last docid of a block, \(d_{i,\min((b+1)B,n_i)-1}\), is its skip target; the block maximum contribution is

\[
M_{i,b}=\max_{r\in \text{block } b} c_i(d_{i,r}),
\]

where \(c_i(d)\) is term \(i\)'s BM25 contribution to document \(d\). Note this is the max of the **contribution**, not of tf: length normalisation varies per document, so only per-row contributions followed by a max form a strict bound.

**skip-AND intersection**: keep a pointer \(p_i\) per list with current docid \(c_i=d_{i,p_i}\). Each round take

\[
\text{cand}=\max_i c_i,
\]

advance every \(c_i<\text{cand}\) to the first row with docid \(\ge \text{cand}\); if any list is exhausted the intersection is empty; if all \(c_i=\text{cand}\) emit cand and advance all by one; otherwise continue with the new max. Correctness: any common docid must be \(\ge\) every current docid, hence \(\ge\text{cand}\), while all skipped rows are \(<\text{cand}\) and cannot be in the intersection. The jump is: if the pointer sits at a block start and that block's last docid \(<\) target, jump to the next block start; otherwise step one row. Pointers only move forward, so the total number of advances is at most \(\sum_i n_i\).

**BM25 contribution**: with \(N\) documents, \( \mathrm{df}_i=n_i \), \( \mathrm{idf}_i=\log(1+(N-n_i+0.5)/(n_i+0.5)) \), \( \mathrm{avgdl}=\frac1N\sum_d |d| \),

\[
c_i(d)=\mathrm{idf}_i\cdot
\frac{f_{i,d}(k_1+1)}
     {f_{i,d}+k_1\left(1-b+b\,|d|/\mathrm{avgdl}\right)}.
\]

In the example \(N=6\), \( |d|=4\), so \(b|d|/\mathrm{avgdl}=0.75\) and the \(k_1\) term is always 1.2. Measured: \( \mathrm{idf}_0=\log(1+0.5/6.5)=0.07410797215372183 \) and \( \mathrm{idf}_1=\mathrm{idf}_2=\log 2.8=1.0296194171811581 \). With the three lists

```
score(d0) = 2.5472445775
score(d1..d4) = 0.0741080
score(d5) = 3.1355985302
```

**Block-Max WAND bounds and pivot**: let \( \mathrm{lo}=\min_i c_i \) be the smallest current docid and

\[
\mathrm{hi}=\min_i \text{end}_i
\]

the smallest current block end. For any \(d\in[\mathrm{lo},\mathrm{hi}]\) and any term \(i\): if \(c_i\le d\), then \(d\) lies inside term \(i\)'s current block (because \(c_i\le d\le \mathrm{hi}\le \text{end}_i\) and \(d\ge c_i\ge\) block start), so its contribution is \( \le M_{i,b_i} \); if \(c_i>d\), term \(i\) has no posting at \(d\) and contributes 0. Hence

\[
\mathrm{score}(d)\le \sum_{i:\,c_i\le d} M_{i,b_i}.
\]

Sort the lists by \(c_i\), accumulate the bounds, and the first \(c_i\) whose prefix sum exceeds the current top-k threshold \(\theta\) is the pivot. Every \(d<\text{pivot}\) has bound sum \(\le\theta\) and can be skipped safely; if no prefix sum up to \( \mathrm{hi} \) exceeds \(\theta\), the whole \([\mathrm{lo},\mathrm{hi}]\) can be skipped. These are exactly the reference's branches:

- `pivot == lo`: fully score lo (only lists with \(c_i=\mathrm{lo}\) contribute), push/replace in the heap, advance every pointer at \(\mathrm{lo}\);
- `pivot > lo`: advance every pointer with \(c_i<\text{pivot}\) to \(\ge\text{pivot}\) using skips, counting wholly skipped blocks in `pruned_blocks`;
- no pivot: advance every pointer with \(c_i\le\mathrm{hi}\) to \(>\mathrm{hi}\), skip the whole range and count whole blocks.

**Why trusting only the current block is safe**: the candidate interval is truncated at \( \mathrm{hi}=\min \text{end}_i\) at the same time. Without that ceiling, a term's current block could end before \(d\), so its block maximum would no longer cover \(d\) and the bound would become too tight, pruning true results. `hi = min(block_end)` guarantees that within the interval every term's current block covers the whole interval.

**Threshold and ties**: the heap keeps the \(k\) best documents with key `(score, -docid)`; the top is the worst (smallest score, and for ties the largest docid). The threshold \(\theta\) is the \(k\)-th best score (0 while fewer than \(k\) are held). A pivot only forms when `acc > theta`, so a document whose score is **exactly** the threshold never enters. This is safe because scoring proceeds in ascending docid: every document already in the heap has a smaller docid than the current lo, and a later tie must lose to an earlier index. Measured:

- In the example with `k=2`, `pruned_blocks=1`. After doc0 (2.5472) and doc1 (0.0741080) are in the heap the threshold is 0.0741080; term0's block `[2,3]` has maximum exactly 0.0741080, cannot strictly exceed the threshold, and is skipped whole. doc2 and doc3 would tie but have larger docids, so they lose anyway.
- In the same example with `k=3`, `pruned_blocks=0`. doc2 is fully scored before the heap fills, so block `[2,3]` is only partially scanned and, by the "whole block" definition, does not count; afterwards the `[4,5]` block's low-doc bound exceeds the threshold at doc4, so the pivot hits doc4 and no whole block is left to prune.
- On the 21-document long-tail case (term0 tf=1 on docs 0..19 and tf=50 on doc20; term1 has postings only on doc0/doc20), measurement gives **9** pruned blocks for both `k=1` and `k=2`, and **8** for `k=3` — the threshold changes with heap content, so the prune count is not a constant.

**The `pruned_blocks` counting convention**: `_advance` counts 1 only when "the pointer is at a block start and the block's last docid is \(<\) target", jumping to the next block start. If the pointer is already inside a block, it walks to the block end and that block counts as partially scanned. Thus no block is counted twice, and "pruned" means exactly "no document in the block was fully scored".

**AND versus top-k**: `candidates` is the boolean AND intersection; the top-k domain is the **union** of all postings (WAND is a weak AND: a document containing only some terms can still enter through the accumulated score). An empty query makes both empty; if any postings list is empty the AND intersection is empty, yet top-k can still score the remaining lists — this is the direct consequence of "query term absent from corpus" and "AND has no solution" being different semantics.

**Empty documents and avgdl=0**: when every `doc_lens` is 0 the length ratio is taken as 0, the denominator becomes \(f+k_1(1-b)\), and contributions stay positive (tf and idf are positive); no NaN appears.

## Key points and common mistakes

1. **docids must be strictly increasing**: duplicates or descending order raise `ValueError`; the implementation advances pointers instead of building an \(N\)-length dense array, which is the essential difference for sparse postings.
2. **The skip target is the block's last docid, not its first**: a whole block may be jumped only when its last docid \(<\) target; jumping at the first row would miss rows inside the block.
3. **A block bound must cover the entire candidate interval**: truncate with \( \mathrm{hi}=\min \text{end}_i\); otherwise, once the current block ends, its maximum no longer bounds later documents and true answers get pruned.
4. **The bound is over contributions, not tf**: BM25 length normalisation makes equal-tf documents contribute differently; only per-row contributions followed by a max are strict.
5. **Threshold ties use strict `>`**: equal-to-threshold documents cannot enter; this relies on "score in ascending docid + smaller index wins ties". The measured difference between `k=2` (1 pruned block) and `k=3` (0) is exactly the interplay of ties and partial scans.
6. **Whole-block counts must not repeat**: a pointer inside a block walks to the end without counting; only a jump from the block start counts 1.
7. **The top-k domain is the union, not the corpus**: documents absent from every postings list score 0 but do not occupy top-k slots; `k>U` returns \(U\), not \(N\).
8. **Two meanings of an empty list**: an empty query empties both `candidates` and `topk`; one empty term only empties the AND intersection while top-k still scores the union of the others.
9. **Repeated terms accumulate per occurrence**: the same list twice doubles the score and leaves candidates unchanged; do not de-duplicate.
10. **`k=0` short-circuits**: no heap and no scoring, `pruned_blocks=0`, but `candidates` must still be computed.
11. **`block_size=1` is legal**: every row is a block, so skip and block bounds degenerate to row-by-row; a short final block is legal too.
12. **Pruning must not change results**: this is a hard requirement and judging runs brute-force full scoring for every case; any too-tight bound will make some topk disagree with brute force.

## Differences from neighboring problems

- **vs `retrieval-rrf-fusion` (same collection)**: RRF only fuses ranks; its input is already-ordered lists and it has no postings, tf, blocks or bounds. This problem builds/consumes inverted lists, uses real-valued BM25 scores and block-bound top-k pruning.
- **vs `retrieval-rag-chunk` (same collection)**: RAG chunking only outputs a "token index matrix"; it is pure splitting with no scoring and no candidate pruning. This problem outputs docids, top-k and a prune count.
- **vs `retrieval-ivf-assign` (same collection)**: IVF is a dense Euclidean argmin over vectors with a full-corpus candidate domain; this is sparse inverted indexing with docid pointers and threshold pruning — different data structures and algorithms.
- **vs `retrieval-bm25-score` (same batch)**: that problem computes df/avgdl from a dense token corpus and returns **dense corpus** scores; this one takes already-built postings, returns only a union top-k, and additionally requires the AND intersection and `pruned_blocks`. Same BM25 formula, but one is the scoring body and the other is inverted-index query execution.
- **vs `searchsorted-bincount`**: that is a "value → bucket + count" histogram primitive with no documents, terms, docid monotonicity or bound pruning.

## Cross-check convention and complexity

- **Primary oracle**: `_oracle` is pure-Python brute force — for every document and every term it scans the full postings and accumulates BM25, then `sorted(union, key=(-score, docid))`; the AND intersection linearly searches every docid in every list. It is structurally different from the reference's "block bounds + pivot + skip pointers" and specifically catches tight bounds, wrong skips and tie-convention bugs.
- **Coverage**: the example (all of `candidates/topk/pruned` filled from measurements), 5 mixed corpora (zero-length docs, `k=100`, `block_size=1/2/3`, `b=0.2/1.0`, an empty postings list), a long skip-AND case (\(N=25\), `block_size=1/2/4/64`, intersection `[12,24]`), empty query, missing term, three fully tied documents, `k=0`, `k>U`, single term, repeated term, and 14 invalid-input groups. Every case compares `candidates` and `topk` with `torch.equal` and asserts that `pruned_blocks` is a non-negative integer.
- **Pruning genuinely happens**: the example gives `pruned_blocks=1` at `k=2`; the 21-document long-tail case measures 9 for `k=1/2` and 8 for `k=3`. A further 300 random small postings sets (\(N\le12\), random block_size) were cross-checked locally with no topk disagreement against brute force, with a maximum of 13 pruned blocks in a single case.
- **Complexity**: block construction \(O(\sum_i n_i)\); skip-AND worst case \(O(\sum_i n_i)\) (monotone pointers); WAND sorts the active lists each iteration in \(O(T\log T)\), fully scores a document in \(O(T)\), and runs for "scored documents + whole blocks skipped" iterations, worst case \(O((U+B)T\log T)\) with \(U\) the union size and \(B\) the block count; heap operations \(O(U\log k)\). Space \(O(\sum_i n_i + k)\). Judging sizes are tiny and far inside the limit.
