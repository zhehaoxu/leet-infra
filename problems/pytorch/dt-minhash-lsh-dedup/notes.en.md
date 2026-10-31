# Reference Approach: Signature, Banding, Verification, Clustering in Series

## Implementation

Each of the four stages consumes only the previous stage's output, so a mistake in one stage
shows up in the next — which is why all five intermediate results are returned.

1. **Shingle fold**: `sid = (sid * 31 + t) % MOD` is bit-by-bit Horner, congruent to the power
   expansion \(\sum_k t_k 31^{n-1-k} \bmod \text{MOD}\) but implemented differently; the result is a
   **set**, so a repeated window does not increase any count.
2. **Signature**: one `min` per entry, `MOD` for an empty set. There is no RNG — the coefficients
   `(a, b)` arrive as arguments, so the signature is a pure function.
3. **Banding**: block `k` is the **contiguous** slice `k*rows:(k+1)*rows`, bucket members are paired,
   and the per-block pairs are unioned, de-duplicated and sorted. Contiguous and strided are two
   different banding schemes with the same `bands * rows` and the same set of positions, yet they can
   disagree on collisions; the derivation below is exactly the data the judge uses to tell them apart.
4. **Verification and clustering**: only nominated pairs are checked, with
   `inter / union >= threshold` (an empty union scores `0.0`); the surviving edges form connected
   components whose label and representative are the component's **smallest index**.

For banding to be distinguishable, the signature must not be a function of `min(S)` alone. With
`MOD = 2**61-1`, coefficients like `a = 2**31` turn the multiplication into a cyclic left shift over
61 bits, so different permutations pick different argmin elements from the same set:

```text
# multiplication by 2**k modulo 2**61-1 is a cyclic left shift on 61 bits,
# because 2**61 = 1 (mod 2**61-1).  With n = 1 the shingle id is the token.
# PERM_ROT = [(2**31,0), (2**32,0), (2**33,0), (2**34,0)]

A = {2**30}            # k = 30
   k+31 = 61 = 0  mod 61 -> 2**0 = 1
   k+32 = 62 = 1  mod 61 -> 2**1 = 2
   k+33 = 63 = 2  mod 61 -> 2**2 = 4
   k+34 = 64 = 3  mod 61 -> 2**3 = 8            sig_A = [1, 2, 4, 8]

B = {2**30, 2**28}     # k = 30 gives 1, 2, 4, 8 as above; k = 28 gives
   k+31 = 59 -> 2**59   (larger, not the min)
   k+32 = 60 -> 2**60   (larger, not the min)
   k+33 = 61 = 0  -> 1  (beats 4)
   k+34 = 62 = 1  -> 2  (beats 8)               sig_B = [1, 2, 1, 2]

contiguous bands, rows = 2:  (1,2) == (1,2) hit ; (4,8) != (1,2) miss
strided    bands, rows = 2:  (1,4) != (1,1) miss ; (2,8) != (2,2) miss
exact Jaccard = 1/2, so the pair is a duplicate only if banding proposed it
```

The reference solution uses union-find to get the components but rewrites every label to the
component's smallest index **before returning**: a union-find root depends on merge order, and
leaking it into the result would make the answer implementation-defined. The two general invariants
`labels[i] <= i` and `labels[labels[i]] == labels[i]` in the judge exist precisely to catch that.

There is no offline implementation to compare against, so the two mathematical properties this
algorithm is *supposed* to satisfy were measured directly: signature agreement should approach the
exact Jaccard, and banding recall should approach \(1-(1-s^r)^b\).

```text
# 140 documents, 40 tokens each, 7 groups of 20 sharing r of their tokens
# (J = r / (80 - r)); 400 random permutations; 9730 pairs.
# signature agreement rate per exact-Jaccard bucket, averaged over pairs:
  exact 0.0000 (8590 pairs)  ->  0.0000
  exact 0.1111 ( 190 pairs)  ->  0.1468
  exact 0.2500 ( 190 pairs)  ->  0.2567
  exact 0.4286 ( 190 pairs)  ->  0.4406
  exact 0.6667 ( 190 pairs)  ->  0.6662
  exact 0.8182 ( 190 pairs)  ->  0.8229
  exact 1.0000 ( 190 pairs)  ->  1.0000
  largest bucket deviation 0.0357 (the 0.1111 bucket, 400-perm sampling noise)

# banding recall over the 190 pairs with exact J = 0.8182, theory 1-(1-s^rows)^bands
  bands  4 rows 10   measured 0.195   theory 0.439   candidate pairs 236
  bands  8 rows  5   measured 1.000   theory 0.974   candidate pairs 524
  bands 10 rows  4   measured 1.000   theory 0.997   candidate pairs 642
  bands 20 rows  2   measured 1.000   theory 1.000   candidate pairs 846
  bands 40 rows  1   measured 1.000   theory 1.000   candidate pairs 1140
  bands  2 rows 20   measured 0.016   theory 0.036   candidate pairs 193
```

The first table tracks the exact value bucket by bucket, and the second shows the dial the statement
advertises: more `rows` means stricter nomination and fewer candidates, more `bands` means the
opposite. Only the `bands=4, rows=10` row sits clearly below theory (0.195 against 0.439), because its
190 pairs come from the same 20 documents, so **the pairs are not independent** — if one document's
signature leans towards its own unique tail, it collides with nobody. The effective sample is the
number of documents, not the number of pairs, which makes the error bar far wider than a binomial
estimate suggests. The judge never uses a statistic: everything is an exact comparison, and randomness
appears only in this self-check.

## Correctness checks

Twelve `check_*` functions, all compared against a structurally independent oracle: the oracle folds
shingles by power expansion, sorts the hashed values and takes the first, finds candidates by scanning
all pairs with an element-by-element band comparison, and clusters by BFS over an adjacency list —
while the reference uses Horner, a running minimum, bucketing and union-find. No intermediate result
is shared. Beyond the 5-tuple comparison, `_verify` enforces: both pair lists hold `tuple`s with
`p < q`, unique and lexicographically ascending; `duplicates` is a subset of `candidates`;
`labels[i] <= i` and `labels[labels[i]] == labels[i]`; `kept == sorted(set(labels))`; a document
shorter than `n` has a whole row equal to `MOD` while every other entry lies in `[0, MOD)`;
and `docs` and `perms` are unmutated.

- `check_example`: hand-computed 5-tuples for both statement examples.
- `check_matches_oracle`: 23 fixtures (20 hand-written plus 3 deterministic LCG corpora) covering an
  empty corpus, one document, all-identical documents, all-empty documents, `num_perm=1`, `bands=1`,
  `rows=num_perm`, `perms` given as lists rather than tuples, repeated permutations, `a = MOD-1`,
  tokens larger than `MOD`, huge tokens, and thresholds 0.0/0.4/0.5/1.0.
- `check_banding_nominates_and_exact_check_disposes`: the rotation signature pair above — `rows=1`
  nominates two pairs and threshold 0.5 rejects both; `rows=2` drops both from the candidate list.
  Nomination and decision are two separate switches.
- `check_true_duplicate_never_patched_in`: a genuine duplicate that banding never nominated must stay
  out of `duplicates`, which kills "just compare all pairs" answers.
- `check_empty_shingle_set_rules`: three windowless documents nominate each other, produce zero
  duplicates above threshold 0 and one whole cluster at `threshold = 0.0` (`0.0 >= 0.0` counts); an
  empty document never shares a bucket with a non-empty one.
- `check_sid_is_folded_modulo`: with `n=1`, `MOD` folds to `0`; with `n=2`, `(MOD+5, 7)` and `(5, 7)`
  both give `162`; `(2*MOD+5, 7)` makes a no-modulus answer produce a different set.
- `check_shingle_set_not_multiset`: `[1,2,1]` and `[1,2,1,2,1]` have the same sid set so `J = 1.0`;
  treating windows as a multiset gives `2/4 = 0.5`, which drops out at threshold 1.0.
- `check_threshold_boundary_is_inclusive`: `2/4 = 0.5` counts as a duplicate and `0.5000001` does not;
  `4999/10000 = 0.4999` is **not** a duplicate at threshold 0.5 (rounding first would flip it); an
  integer `threshold` of 1 behaves as 1.0.
- `check_clusters_are_transitive`: `0~1` and `1~2` pass while `0~2` is merely a candidate, yet all
  three land in one cluster; and when the only duplicate edge sits on late indices the representative
  must still be the smallest index.
- `check_bands_are_contiguous_slices`: with `sig_A = [1,2,4,8]` and `sig_B = [1,2,1,2]` contiguous
  banding nominates, strided banding does not, and `rows=4` nominates nothing at all.
- `check_scale`: a 300-document LCG corpus with 8 permutations under four `(bands, rows)` splits, plus
  two 300-token documents.
- `check_invalid`: 38 rejected inputs plus 4 legal boundary calls. The five threshold cases must use
  `bands=1, rows=1`, otherwise `bands * rows != num_perm` fires first and silently turns the threshold
  validation into a pass — this version of the test had exactly that bug.

Mutation testing ran 28 mutants and all 28 were caught: no modulus in the fold, base 32, little-endian
fold, multiset Jaccard, `max` instead of `min`, first window instead of minimum, swapped `a`/`b`,
permutation without modulus, empty-set signature filled with 0 and left empty, strided bands, one row
short per band, candidates not merged across bands, candidates unsorted, skipping LSH to compare every
pair, `>` instead of `>=`, union size as a plain sum, empty union scored 1.0, cross-multiplication
instead of division, rounding before the comparison, labels taken from the union-find root, labels as
the component maximum, `kept` computed by dropping the first element of every pair, no clustering at
all, and three loosened validators (skipping the `bands*rows` check, clipping negative tokens to 0,
clamping the threshold into `[0,1]`). The last three are caught because the judge asserts that
`ValueError` is raised, not what comes back after it.

## Scope and sources

- **Nothing offline to compare against**: `.venv-eval` has no `datasketch`, `sklearn`, `scipy`,
  `xxhash`, `mmh3` or `datatrove`, and no source for them either. So this problem **claims no
  attribution** and reproduces no library's constants or API: `BASE = 31`, `MOD = 2**61-1`, `MOD` as
  the empty-set fill, contiguous banding, de-duplicated candidates across bands, the float rule
  `inter / union >= threshold`, and labels equal to the component minimum are all **defined by this
  problem**. Real pipelines differ at every one of these points (signature length, how bands and rows
  are cut, whether b-bit compression is used, which hash family); the problem fixes one coherent
  convention instead.
- **The properties are backed by a local measurement**: the two tables above come from running
  `random.Random(20261027)` over 140 documents with 400 permutations and 9730 pairs, confirming that
  signature agreement tracks Jaccard and that banding recall tracks \(1-(1-s^r)^b\) under these
  constants. The judge itself uses no statistic — every assertion is deterministic.
- **Division of labour with `eval-shingle-contamination` (order 300)**: that problem computes the dense
  shingle **containment and Jaccard matrices** of an evaluation set against a training set, and its
  subject is which side the denominator sits on plus the flagged indices — small enough to intersect
  every pair directly. This problem is about the **scalable approximate-retrieval skeleton**: fixed-length
  signatures, band buckets, candidate nomination, exact verification, transitive clustering and a keep
  list, where Jaccard is only one predicate in the middle. The two also fold shingles differently — 300
  keeps id tuples, this one folds each window into a single integer modulo `MOD`.
