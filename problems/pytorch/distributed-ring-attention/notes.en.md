# Reference Approach: Ring, Online Softmax and the Communication Bill on One Schedule

## Implementation

Five stages: validate the arguments, replay the C steps rank by rank, decide per step whether to
skip, run the online softmax recurrence row by row, then normalise. `traces`, `sources`, `skipped`
and `comm_bytes` are all returned because a single misread step has to surface somewhere: looking at
`out` alone, even a ring that turns the wrong way can occasionally produce the same weighted sum.

1. **Direction of the ring.** In step s, rank r receives src = (r - s) mod C. Step 0 is the rank's
   own block; every later step moves one rank upstream, so after C steps each block has been seen
   exactly once.
2. **Whole-block skipping.** When `causal` is true and src > r, every key of that block sits at a
   global position > r*S + S - 1, so the block is entirely invisible: the `if not dead` branch is
   never entered, not a single exp is evaluated, and m, l and acc keep their values.
3. **The recurrence.** corr = exp(m - m'), and l and acc are multiplied by corr before the new block
   is added. On the first block m is -inf and `exp(-inf - finite)` is 0.0 in Python, which turns
   "nothing accumulated yet" into a multiplication by zero, no branch needed.
4. **The denominator cannot be zero.** m' is always the logit of some key inside the block, and that
   key contributes exp(0) = 1 to l, so l >= 1 after every step and the final acc / l is safe.
5. **The communication bill.** Each step moves one KV block in and one out for
   2*S*D*dtype_bytes; after the last step there is no downstream hop, so it is 0. Skipping saves
   arithmetic only: a block that is swallowed is still received and still forwarded.

Two rewrites are worth recording, because they decide how strictly the statement may be worded:

- src > r is the same as s > r under src = (r - s) mod C: for s <= r, src = r - s <= r; for s > r,
  src = C - (s - r) >= r + 1. So writing the skip test as "the source index is out of reach" or as
  "the step index is out of reach" gives bit-identical output, which is why the statement pins the
  output and not the form of the test.
- `corr = exp(m - m')` must **not** be rearranged into `1 / exp(m' - m)`: on the first block
  m = -inf, so m' - m = +inf, and `math.exp(inf)` raises `OverflowError` in CPython instead of
  returning inf. That is not an equivalent rewrite but a real bug, and mutation testing catches it
  through `check_large_logits`.

## Correctness checks

Thirteen `check_*` functions, with a structurally independent oracle for the main comparison: it
first concatenates all shards into one global (N x D) matrix, computes the output with an **offline
softmax** (row maximum first, one single sum afterwards, no corr anywhere), recomputes `traces`
**from scratch** over every key visible so far, builds `sources` as
`range(r, -1, -1) + range(C-1, r, -1)`, decides `skipped` by the position test
src*S > r*S + S - 1, and builds `comm` as a list padded with a trailing zero. The reference is a
per-rank, per-step online recurrence. The two paths share no intermediate result.

- `check_example`: all four statement examples compared cell by cell, including `traces` and the
  copied-over state of skipped steps.
- `check_matches_oracle`: 18 fixtures (one rank up to eight, S from 1 to 6, D from 1 to 3, causal and
  non-causal, scale = 1/sqrt(D)) plus three random cases crossed with three `dtype_bytes` values.
- `check_matches_dense_attention`: a second independent route, using `torch` to build the global
  score matrix, `masked_fill(-inf)`, `softmax` and multiply by V, compared at a 1e-12 relative
  tolerance.
- `check_sources_rotate_backwards`: `sources[r][0] == r`, consecutive steps differ by -1 mod C, and
  every row is a permutation; with C=1, `sources == [[0]]`.
- `check_causal_skip_is_whole_block`: `skipped[r][s]` must equal the position test, and nothing is
  skipped when the mask is off.
- `check_uniform_logits_count_visible_keys`: with q all zeros, l must equal the number of visible
  keys (p+1 when causal, N otherwise) and the output is the running mean of the visible v. This is
  what pins "the mask only knows global positions".
- `check_first_visible_key_survives`: when causal, global row 0 has a single visible key, so the
  output must equal v[0] **bit for bit** (the weight is exactly 1, no room for float noise).
- `check_skipped_step_keeps_state`: a skipped step repeats the previous `traces` bit for bit and is
  still charged; with C=4 exactly 0+1+2+3 = 6 blocks must die; causal and non-causal outputs must
  differ.
- `check_comm_account`: for four (C, S, D, dtype) combinations the bill equals the "payload list with
  a trailing zero".
- `check_large_logits`: q=20 with k=20 gives a logit of 400 (exp without subtracting the maximum
  overflows), q=-20 gives -400 (the running max must be allowed to be negative), and both must come
  out right; two more random cases with entries at +-50 and scale=30 must stay finite throughout.
- `check_single_rank`: C=1 degenerates to ordinary causal attention and is compared against a
  hand-written lower-triangular softmax.
- `check_scale`: 8 ranks x 8 rows x 4 dims, a single rank of 16 rows x 8 dims, 4 ranks x 16 rows x
  8 dims, plus an all-zero 16-row ladder.
- `check_invalid`: 39 rejected inputs and 4 legal boundaries, covering the shard structure, the entry
  types and all three scalar arguments; the `bool` trap (it is both an `int` and a plausible 0/1) is
  blocked once each in `causal`, `dtype_bytes` and the entries.

Mutation testing ran 36 mutants and all 36 were caught, with two further rewrites documented as
equivalent. One of them deserves a note: replacing `dead = causal and source > rank` with
`dead = causal and step > rank` produces **bit-identical output**, because the identity above makes
the two tests the same thing. A rewrite that cannot change the output must not be judged wrong,
which is why the statement pins the output rather than the form of the test. Conversely,
`corr = 1 / exp(m' - m)` looks like a mere rearrangement but raises `OverflowError` on the very
first block, so it is a genuine error that has to be caught.

What skipping is worth can simply be counted (first table of `_xcheck554.py`, S=8, D=4, dtype=4):

```text
# C ranks, shard length 8, head width 4, causal=True, dtype_bytes=4
# tile pairs = score entries an implementation would touch if every received block were scored
   C  blocks  skipped  share | score-tile pairs skip / no-skip   saved | exp calls | comm bytes
   2       4        1   0.25 |         192 /       256    0.25 |       160 |        512
   4      16        6   0.38 |         640 /      1024    0.38 |       608 |       3072
   8      64       28   0.44 |        2304 /      4096    0.44 |      2368 |      14336
  16     256      120   0.47 |        8704 /     16384    0.47 |      9344 |      61440
```

The number of dead blocks is always C(C-1)/2, a share of (C-1)/(2C): at large C nearly half of the
blocks disappear from the arithmetic while the comm bytes do not drop by a single byte, which is the
quantitative version of "saves compute, not bandwidth". Note that the baseline has to deal with a
block that leaves no key at all (`max()` over an empty sequence raises `ValueError`); in other words,
**without whole-block skipping one has to test each row for emptiness**, and this problem moves that
test up to the block level.

The necessity of the online form can be measured the same way: the naive formulation that never
subtracts the maximum collapses once the logits grow.

```text
# 4 ranks, shard length 8, head width 4, entries in [-1, 1], 20 random trials
# blown = rows where the naive exp(logit) form overflows or divides 0 by 0
  scale   max|logit|   naive rows blown up   max|online - naive| (surviving)
    1.0          3.3           0/640         2.50e-16
    5.0         16.5           0/640         7.29e-16
   20.0         66.1           0/640         2.86e-15
   50.0        165.4           0/640         3.66e-15
  200.0        661.4           0/640         1.13e-14
  500.0       1653.6         134/640         2.43e-14
 2000.0       6614.5         596/640         1.26e-04
```

From scale=500 on the naive form starts raising `OverflowError` (exp(1653) is already beyond double
precision), and at scale=2000 596 of the 640 rows cannot be computed at all; the rows that survive
have lost four significant digits to cancellation (1.26e-04). The online form keeps every exponent
at or below zero, holds the error at the 1e-14 level, and l >= 1 rules out a division by zero.

## Scope and sources

- **No attribution claimed.** Ring Attention is an algorithm by Liu et al. (2023), and open
  implementations such as `ring-flash-attention` do pass KV blocks around a ring, but there is no
  implementation to compare against offline (`.venv-eval` has only torch / numpy / transformers).
  The source order `src = (r - s) mod C`, the skip test, "comm is charged only C-1 times", "a
  skipped step copies the previous traces" and "the caller supplies `scale` instead of 1/sqrt(d)"
  are all **defined by this problem**. Real systems differ at every one of these points: KV rotates
  per head or per layer, zigzag index orders balance the causal load, the block size can be
  independent of the rank count, and communication overlaps with compute.
- **The properties are backed by a local measurement.** The skipped share (C-1)/(2C) is a
  combinatorial identity (C(C-1)/2 dead blocks out of C^2), and the script verifies it for
  C = 2, 4, 8, 16 together with the claim that skipping and not skipping give **bit-identical**
  output.
- **Division of labour with neighbours**: `flash-attention-tile` is single-GPU tiled online softmax
  with no ring, no communication bill and no whole-block skipping; `distributed-sequence-parallel`
  computes closed-form memory and communication bytes without replaying a single step;
  `distributed-ring-allreduce-steps` snapshots the steps of ring all-reduce, which has nothing to do
  with attention; `distributed-all-to-all` and `distributed-tp-split-rearrange` take the Ulysses
  all-to-all route rather than a ring walk. This problem fills the "ring rotation + online
  recurrence + whole-block causal skipping" cell.
