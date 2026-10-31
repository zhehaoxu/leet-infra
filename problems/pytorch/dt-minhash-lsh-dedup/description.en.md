# MinHash Signatures and LSH Banding for Near-Duplicate Dedup

## Task

Deduplicating a corpus cannot afford pairwise set comparisons: 100k documents are 5 billion pairs, each needing an intersection. Production pipelines split "who looks like whom" into two steps. First **MinHash** compresses each document into a fixed-length signature — under one permutation, the probability that two documents share a signature entry is exactly their Jaccard similarity. Then **LSH banding** turns "find similar pairs" into "find bucket neighbours": cut the signature into `bands` blocks of `rows` entries each, put documents with an identical block into the same bucket, and take the pairs inside each bucket as candidates. Every candidate is then re-checked **exactly** against the original shingle sets. The band/row split is the recall-versus-candidate-count dial — larger `rows` is stricter and proposes fewer pairs, larger `bands` is looser. This is the skeleton behind the minhash-dedupe stage of GPT/LLaMA-style corpora (CC-Net, Dolma, RedPajama).

This problem walks the whole skeleton: signature, banding, candidate pairs, exact verification, connected-component clustering and the keep list — all five intermediate results are returned, so a mistake anywhere shows up downstream. To keep judging deterministic, **the permutation coefficients come in as an argument**: no RNG, no tokenizer, no hashing library, pure Python integer arithmetic.

## Input

- `docs`: `list[list[int]]`, each document a sequence of non-negative token ids (already split). The number of documents and every document length may be 0.
- `n`: positive integer, shingle width (length of the contiguous window).
- `perms`: non-empty `list` of length `num_perm`, each item a pair `(a, b)` (either a `tuple` or a `list`) holding one permutation's coefficients.
- `bands`, `rows`: positive integers with `bands * rows == num_perm`.
- `threshold`: float in `0.0 <= threshold <= 1.0`, the similarity cutoff used when verifying candidates.

Calling any ready-made MinHash / LSH / dedup / similarity library is **forbidden**; signatures, buckets and the union-find must be hand-written.

## Exact Rules

Let `BASE = 31` and `MOD = 2305843009213693951` (that is \(2^{61}-1\)); all integer arithmetic below is taken modulo `MOD`.

1. **Shingle set.** For a document `d` take every contiguous window of length `n` (overlapping, `max(0, len(d) - n + 1)` of them). Fold a window `w` into one integer:

   `sid = 0`; then for each `t` in `w`, `sid = (sid * 31 + t) % MOD`.

   \(S(d)\) is the **set** of those sid values — the same sid produced by several windows counts once. If `len(d) < n`, then \(S(d)=\varnothing\).

2. **Signature.** Permutation `p = (a, b)` defines \(h_p(x) = (a \cdot x + b) \bmod \text{MOD}\) with range \([0, \text{MOD})\).

   `sig[i][p] = min(h_p(x) for x in S(docs[i]))`; if \(S(d_i)=\varnothing\) then **every** entry of that document is `MOD` (the identity element for `min`, deliberately outside the legal range so empty documents stay recognisable). Signatures follow input document order, shape \((N, \text{num\_perm})\).

3. **Banding.** Block `k` (`0 <= k < bands`) is the **contiguous** slice `sig[i][k*rows : (k+1)*rows]` — not a strided sample. Documents whose slice is identical form one bucket, and every pair of bucket members is a candidate of that block. The overall candidate list is the **union with duplicates removed**: a pair that collides in several blocks is still listed once. Each pair is written `(p, q)` with `p < q`, and the list is sorted ascending lexicographically.

4. **Exact verification.** For each candidate pair

   \[ J(p,q) = \frac{|S(d_p)\cap S(d_q)|}{|S(d_p)\cup S(d_q)|}, \qquad \text{an empty union gives } 0.0 \]

   The decision uses Python float division: a pair is a duplicate when `J(p, q) >= threshold` (**equality counts as duplicate**). Passing pairs go to `duplicates` under the same `(p, q)`, de-duplicated, lexicographic rules. `duplicates` must be a subset of `candidates` — a pair that banding never proposed stays out even if it is a genuine duplicate.

5. **Clustering and the keep list.** Take connected components of the undirected graph whose edges are `duplicates` (transitive closure: `0~1` and `1~2` puts `{0,1,2}` in one component; an unpaired document is its own component). `labels[i]` is the **smallest index** of \(i\)'s component and `kept` is the ascending list of component representatives, i.e. `sorted(set(labels))`. The documents that get dropped are exactly those missing from `kept`.

## Return

A 5-tuple `(sig, candidates, duplicates, labels, kept)`, all plain Python built-ins:

- `sig`: `list[list[int]]`, shape \((N, \text{num\_perm})\); entries of an empty-shingle document are `MOD`.
- `candidates`: `list[tuple[int, int]]`, de-duplicated, `p < q`, ascending lexicographic.
- `duplicates`: `list[tuple[int, int]]`, a subset of `candidates`, same ordering rules.
- `labels`: `list[int]` of length \(N\), holding each component's smallest index.
- `kept`: `list[int]`, ascending.

Any illegal argument raises `ValueError`: `docs` not a list, an inner item not a list, an element that is not an `int` (`bool` does not count) or is negative; `n` not an `int` or < 1; `perms` empty, not a list, an item of length other than 2, a coefficient that is not an `int`, `a` outside \([1, \text{MOD})\), `b` outside \([0, \text{MOD})\); `bands` or `rows` not an `int` or < 1; `bands * rows != num_perm`; `threshold` neither `int` nor `float` (`bool` does not count) or outside \([0, 1]\). \(N=0\) is legal and returns empty `sig`/`candidates`/`duplicates`/`labels`/`kept`. Do not mutate `docs` or `perms`.

## Examples

```
docs = [[1, 2, 3], [1, 2, 3, 2, 3], [4, 5], [7]]
n = 2, perms = [(1, 0), (3, 7), (5, 1), (2, 3)], bands = 2, rows = 2, threshold = 0.5

# shingle sets, sid = Horner with base 31 modulo 2**61-1
S0 = {33, 65}          # (1,2)=33   (2,3)=65
S1 = {33, 65, 95}      # windows (1,2) (2,3) (3,2) (2,3); the repeated (2,3) counts once
S2 = {129}             # (4,5)=129
S3 = set()             # len 1 < n

sig = [[33, 106, 166, 69],
       [33, 106, 166, 69],
       [129, 394, 646, 261],
       [M, M, M, M]]            M = 2305843009213693951, the empty-set sentinel

# rows = 2, so band0 = sig[0:2] and band1 = sig[2:4] (contiguous, not strided)
band0: (33, 106) -> {0, 1}    (129, 394) -> {2}    (M, M) -> {3}
band1: (166, 69) -> {0, 1}    (646, 261) -> {2}    (M, M) -> {3}

candidates = [(0, 1)]          # {0, 1} collides in both bands but counts once
J(0, 1) = |{33, 65}| / |{33, 65, 95}| = 2/3 = 0.6666666666666666 >= 0.5
duplicates = [(0, 1)]
labels = [0, 0, 2, 3]
kept   = [0, 2, 3]             # doc 1 is dropped, doc 3 survives alone
```

Banding only nominates; the exact Jaccard decides. With `rows = 1` the nomination gets visibly looser:

```
docs = [[1, 2, 3], [1, 4, 5], [2, 3, 6], [1, 2, 3], [], []]
n = 1, perms = [(1, 0), (2, 0)], bands = 2, rows = 1, threshold = 0.7

S0 = {1, 2, 3}   S1 = {1, 4, 5}   S2 = {2, 3, 6}   S3 = {1, 2, 3}   S4 = S5 = set()
sig = [[1, 2], [1, 2], [2, 4], [1, 2], [M, M], [M, M]]

band0 (sig[0]): 1 -> {0, 1, 3}    2 -> {2}    M -> {4, 5}
band1 (sig[1]): 2 -> {0, 1, 3}    4 -> {2}    M -> {4, 5}

candidates = [(0, 1), (0, 3), (1, 3), (4, 5)]
J(0, 1) = 1/5 = 0.2  <  0.7   rejected
J(0, 3) = 3/3 = 1.0  >= 0.7   duplicate
J(1, 3) = 1/5 = 0.2  <  0.7   rejected
J(4, 5) = 0.0 (union is empty)  <  0.7   rejected
duplicates = [(0, 3)]
labels = [0, 1, 2, 0, 4, 5]
kept   = [0, 1, 2, 4, 5]       # only doc 3 is dropped
```
