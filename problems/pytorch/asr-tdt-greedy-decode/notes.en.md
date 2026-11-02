# notes — asr-tdt-greedy-decode (order 717)

## Source

- Authoritative source: NVIDIA/NeMo `nemo/collections/asr/parts/submodules/transducer_decoding/tdt_label_looping.py` (GreedyTDTInfer.torch_impl); paper arXiv 2304.06795 (TDT: Token-and-Duration Transducer).
- **Blank forces duration >= 1**: `durations = durations.masked_fill_((durations == 0) & blank_mask, 1)` — applied to blank only; a non-blank 0 duration passes through untouched (into the max_symbols branch).
- **Frame-skip semantics**: after a non-blank emission, `time_indices += duration`, with no RNN-T trailing blank; the official torch_impl carries the comment *NB: difference between RNN-T and TDT* (RNN-T waits in place for a blank after emitting). Duration may push time past the utterance end — the emission ledger snapshots before the jump (time_indices_current_labels), so the "jump out of bounds" emission still counts.
- **Forced-blank advance amount**: `time_indices += 1` — advances exactly 1 frame, ignoring the predicted duration. Not "duration=1" semantics (the difference from duration=1 shows in the forced_blanks ledger and which frame's logits get consumed next).
- **No official anti-loop protection**: a non-blank with duration=0 and max_symbols=None loops forever in torch_impl. This problem adds a ValueError domain constraint; the solution also adds a defensive step cap `max_steps = 100*(T+1)` (absent in the official code, triggered only by pathological inputs, to keep the test harness alive).
- Statelessness: the predictor (StatelessTransducerPredictor) is not modeled — the torch_impl label loop recomputes the joint every step, and under the stateless config (conformer_tdt_bpe_stateless) predictor state does not depend on history, so per-frame explicit inputs are equivalent.

## Test Design

- Oracle = scalar transcription of torch_impl; all five keys compared exactly as integers.
- Pinned-argmax construction: `_rows(spec, T, width)` makes each frame's winner exactly the spec index, everything else -9 — **rows must be written out explicitly frame by frame; reusing spec cyclically via `spec[t % len(spec)]` does not match hand-computed expectations (first version got bitten)**.
- max_symbols=1 semantic anchor: blank is forced right after every non-blank -> forced_blanks == number of non-blanks; distinguished from max_symbols=2 (two consecutive emissions allowed before forcing).
- Pathological input (non-blank d0 + max_symbols=None) asserts ValueError — the hook for the domain-no-guard mutation.

## Mutations (10/10 caught)

- blank-force-off (drop the masked_fill): a d0 blank stays in place. **Note: with scalar blank d0 this mutation necessarily loops forever; it is exposed by the max_steps cap rather than by an assertion difference**.
- nonblank-no-advance / force-adds-duration (forced blank changed to `t += dur`) / lasts-no-reset (lasts not reset on a new timestamp) / emit-after-advance (timestamp taken after the jump) / token-argmax-last (ties take the last index) / emit-blanks (blank emitted) / domain-no-guard / final-clamped (min(T, ·)) / terminal-lt-le (`<` vs `<=` out-of-bounds termination).
- **Infinite-loop mutation lesson**: with a mutation harness that has no timeout, `while True:`-style mutations SIGKILL the whole process (exit 137). Two sources: ① mutations replacing `t += dur` with a dwelling form other than `t += 1`; ② blank d0 itself unprotected. The solution's max_steps cap makes these mutations testable (the over-limit RuntimeError is decidable against oracle behavior) instead of hanging.

## Pitfalls

- Cyclic `_rows` construction via `t % len(spec)` mismatched hand computations -> switched to explicit per-frame rows.
- max_symbols=1 semantics: not "at most 1 non-blank then finish", but "force blank after every non-blank".
- Boundary with 700 (RNN-T greedy) / 699 (transducer loss): this problem owns the TDT dual-head frame-skip ledger; the asr collection now spans Whisper (DTW alignment / language ID) + FunASR (VAD) + NeMo (TDT).
