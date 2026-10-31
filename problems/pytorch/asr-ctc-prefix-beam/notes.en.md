# notes — asr-ctc-prefix-beam (order 670)

## Source

- Origin: the prefix beam search DP from Graves et al. 1504.08129 "Towards End-to-End Speech Recognition with Recurrent Neural Networks" (as transcribed by distill.pub/ctc; common to wav2vec2/icefall). This problem takes the minimal **no-LM, linear-probability** form.
- Three semantic cruxes:
  - **the asymmetric repeated-character branch**: when s ends in c, `s+c` can only extend from p_b (a blank separates two identical characters) while p_nb extends on s — this is the entire substance separating prefix beam search from path beam search;
  - **the blank branch uses the (p_b + p_nb) sum**: after a blank both end-states merge into the same prefix;
  - **pruning and tie-break**: total probability descending, ties by prefix lexicographic order ascending — without the tie-break, tie order depends on dict insertion order (not reproducible).
- Equivalent mutant: initial `(1.0, 0.0)` → `(0.0, 1.0)` is unobservable (the first frame only consumes the p_b+p_nb sum, and the empty prefix's split is rewritten by the first frame). Replaced with the blank-double-expansion mutant.
- Boundary with `ctc-loss` (loss side, taken) and `ocr-ctc-greedy-decode` (greedy path ledger): this is the **prefix-level beam DP**.

## Judging

- solve args: probs [T,V] (linear probabilities), beam_width, blank_id, k (None → beam_width). Returns hypotheses (prefix list + prob).
- Oracle: **exhaustive enumeration of all V^T paths summed by collapse** (a semantic path fully independent of the beam DP), 120 random cases (T≤4, V≤3, softmax-normalized with spikes, bw=999 so no pruning), 1e-9 tolerance; plus hand-computed anchors pinning pruning behavior (beam_width=1 makes (1,) absorb its own blank probability 0.32 vs 0.38 unpruned).
- Anchors: single-frame three-prefix ordering; A blank A merging into one prefix (T=3 hand-computed 0.262/0.729/0.009); AA repetition P(≥1 A)=0.75; lexicographic ties; k truncation; all-blank empty output; zero-probability filtering.
- Invalid inputs: empty/ragged/negative/NaN/bool probs, out-of-range/bool blank_id, non-positive-int or bool beam_width/k → ValueError.

## Mutation log (9/9 caught + 1 equivalent)

repeat-from-both (repetition branch mixes in p_nb), no-repeat-extend (in-place extension lost), blank-from-pnb-only (blank branch drops p_nb), no-prune, sort-ascending, k-plus-one, prob-no-blank (output probability drops p_b), zero-kept (zero-probability hypotheses kept), blank-also-nonblank (blank class also takes the non-blank branch).

**Equivalent mutant**: initial (1.0,0.0) → (0.0,1.0), noted in notes.

## Red lines

No LM scoring (no external score input; an LM variant would need a new problem with dedup checks); no real models/audio; boundary with the `ocr` greedy ledger: this is the **beam DP**.
