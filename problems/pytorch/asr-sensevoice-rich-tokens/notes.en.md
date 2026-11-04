# Pitfalls (order 829)

## Sourcing notes

- Repo `FunAudioLLM/SenseVoice`, `model.py` (SenseVoiceSmall). The official python side has **no** rich_transcription_postprocess source (runtime only ships llama.cpp export and detok) — that post-processing lives in the funasr package / community code and cannot be pinned precisely, so it is excluded (same verdict as QUITO: parts of the official code that are not self-consistent are not made into problems).
- The prefix order is easy to misread: official inference does `cat(textnorm_query, speech)` first, then `cat(input_query, speech)`; after nesting the prefix is **[lid, event, emo, textnorm]**, not [textnorm, lid, ...]. Only careful line-by-line reading of the two cats pins it.
- The training-time `torch.rand(1) > 0.2` lid sampling (20% fallback to auto) is RNG — the judge only covers the deterministic inference-side mapping.
- The embed table is a small 16-row table (rows 0..15), separate from the CTC vocab's big ids like 24884/25016, joined through dict mappings (lid_int_dict). Never mix the two id spaces.

## Test-design pitfalls

1. **LabelSmoothingLoss .float()**: the official implementation casts logits to float (fp32) internally, which collides with fp64 judging ("Float did not match Double") during element-wise cross-checks. The rebuild drops the cast and keeps the input dtype.
2. **ctc_shift direction**: the first draft wrote the content window as [4, L-4) (misreading `lens - 4` as an absolute end coordinate) — in fact `encoder_out[:, 4:]` with `lens - 4` denotes the content frame count, so the window is [4, L). Pinned by asserting 8 content frames for L=12.
3. **ban_emo changed true equivalence**: `(before != after)` equals `(before == emo_unk)` numerically (masking touches no other column) — the mutant `changed = (before != after)` surviving is a true equivalence; replaced with a "mask the wrong column" mutant.
4. **_th_accuracy and _label_smoothing_loss share the identical keep line** — mut anchors must include a following line to disambiguate (the first version was blocked by the count==2 assert).
5. unique_consecutive CTC semantics: only **adjacent** equal ids collapse; non-adjacent equal ids stay (pinned with [3,4,3]); blanks participate in the collapse and are filtered afterwards.

## Mutation design (14 mutants, all CAUGHT)

- Prefix family: wrong order / event-emo row swap / wrong n_added — per-frame embedding-row assertions;
- rich CE family: ignore dropped (denominator pollution) / smooth term gather into nll / ε weights swapped / acc denominator includes ignore — three-way cross-checks at ε=0, ε=1 and the ε mixture, plus the all-ignored boundary;
- greedy family: no unique_consecutive / blank filter dropped — the sequence [3,3,0,0,3,5,5,0,7] pins collapse and filtering separately;
- ban_emo family: -inf → -1 / wrong masked column — -inf assertion + argmax-never-unk + the changed mask, three layers;
- ctc_shift family: start 0 / end minus 4 / missing max clamp — the [12,4,5,3] four-boundary case.
