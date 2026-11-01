# notes — asr-cif-fire-emit (order 694)

## Source

- Official code: modelscope/FunASR `funasr/models/paraformer/cif_predictor.py`,
  module-level `cif(hidden, alphas, threshold)` and
  `CifPredictor.tail_process_fn` (GitHub `main`, fetched 2026-11). Paper:
  CIF 1905.11235.
- Directory exploration: FunASR has no standalone predictor package;
  `CifPredictor` (paraformer family), `CifPredictorV2` (bicif, dual fire
  thresholds), `CifPredictorV3` (cnn variant) — this problem uses the
  classic CifPredictor plus the module-level cif(). The upstream chain
  (conv1d smoothing → sigmoid → smooth_factor/noise_threshold → alphas) is
  out of scope; alphas are explicit inputs.
- Call semantics (forward, target_length=None): alphas → tail_process_fn
  (append tail frame + zero hidden row, token_num=floor(sum)) → cif() →
  slice [:token_num]. For a single sequence the pad/slice steps are
  unobservable, so the interface emits embeds directly.

## Ledger rules

1. **Closed-interval fire** `integrate >= threshold`: exactly 1.0 fires
   (anchors [0.2]*5 and single-frame [1.0]).
2. **cur = gap**: on fire `cur = 1 - integrate_before`, i.e. when the
   cumsum crosses integer k the weight is `k - c_{t-1}`;
   `remainds = c_t - k` starts the next token. Non-fire frames use the
   whole alpha as cur.
3. **Snapshot before decrement**: `list_fires.append(integrate)` sits after
   `integrate += alpha` and before the -1. Using the post-decrement value
   loses exactly-at-threshold emissions (snapshot 0.0) — observable via the
   `snapshot-removed` mutation (fire_positions permanently empty).
4. **frame semantics**: non-fire frames do `frame += alpha*h_t`; a firing
   frame first adds the gap part, emits, then resets to `remainds*h_t`. An
   emitted embedding is the completed weighted sum at firing time.
5. **Tail mechanism**: `tail_process_fn` (mask=None branch) appends one
   `tail_threshold` frame to alphas and one zero-vector row to hidden,
   **with frame alignment**: the tail frame (index T) uses exactly the
   appended zero row. The gap weight multiplies zeros, so the final token's
   embedding accumulates only from real frames. The official invariant
   `len(hidden) == len(alphas)` is the most important input constraint —
   the first test draft fed hidden with more frames than alphas, misaligning
   the tail frame onto a real frame; the assertions caught it and the
   invariant is now pinned in the starter.
6. **token_num dual conventions**: use_tail → `floor(sum_ext)` (returned
   by tail_process_fn); no tail → `round(sum)` (len_labels inside cif()).
   For single sequences use_tail always satisfies
   `emission count == floor(sum_ext)` (induction: post-fire remainder < 1);
   without tail the emission count can be < round(sum) (anchor [0.6]*3:
   1 vs 2). In batch use len_labels also drives the pad width and forward
   then slices [:max(floor)] — unobservable for single sequences, not
   tested.

## Oracle design

Analytic independent implementation: cumsum c_t; frame t contributes the
half-open interval [c_{t-1}, c_t); token k's weight is the intersection
length with [k-1, k); the firing frame is the first t with c_t >= k. No
state machine. Random cross-checks use an **m/256 grid** (power-of-two
denominators, exact floats, no drift, no boundary flips) on 300 cases,
plus assertions of the use_tail identity and strictly increasing
fire_positions.

## Mutation record

All 13 caught (round one had 1; the fixed rerun 13 — the first draft of
`snapshot-after-decrement` truncated the if/else and raised
IndentationError, replaced by `snapshot-removed` deleting the snapshot line,
which makes fire_positions permanently empty). Coverage: fire closed/open
intervals (both decision sites), snapshot existence and timing, cur as gap
vs whole alpha, gap-computation timing, remainder reset/zero/no-reset,
integrate without decrement, tail not appended / missing zero row,
token_num floor↔round swaps, embeds reversed, remainds untracked. No
equivalent mutations survived.

## Lessons

- Three anchor hand-math errors early on (the per-frame fire sequence of
  [0.9]*6, the tail zero-row alignment, mismatched hidden/alpha counts),
  all located by printing actual values; `0.9` on consecutive frames fires
  every frame (post-fire remainder 0.8 + 0.9 = 1.7 ≥ 1) is the most
  counter-intuitive one.
- The tail zero-row alignment only becomes self-consistent once the
  `len(hidden) == len(alphas)` invariant holds — the input constraint
  itself is part of the ledger.

## Dedup

- The todos entry ⑤ originally said "tail remainder distributed
  proportionally" — the official code has no such distribution; it is a
  tail_threshold appended frame. Corrected per the source (todos updated).
- Different from `ctc-loss` (loss-side DP) and RNN-T (candidate ④, forward
  DP): CIF is an inference-side weighted-accumulation state machine.
  Unrelated to `spec-decoding`.
- FunASR's CifPredictorV2 (bicif) dual thresholds are not covered;
  `gen_frame_alignments` (batch alignment matrices) is convoluted and
  meaningless for single sequences, not covered.
