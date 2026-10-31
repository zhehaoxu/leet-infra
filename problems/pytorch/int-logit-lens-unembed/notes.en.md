# notes — int-logit-lens-unembed (order 661)

## Source

- Origin: nostalgebraist's LessWrong post "interpreting GPT: the logit lens" (2020); no standalone official repo (transformers has none), so the problem follows the simplified "final norm + unembed as explicit inputs" formulation mandated by the todos #103 red line.
- Boundary with 653 `int-tuned-lens-affine`: the tuned lens is a **learned affine probe** (per-layer W_l/b_l, no final LN); this one is the **training-free readout** (reusing the final LN), with an agree/rank ledger instead of CE/KL.

## Judging surface

- solve args: hiddens `[L,D]`, lnf_w/lnf_b `[D]`, unembed `[D,V]`, apply_norm bool, eps>0 (default 1e-5). Returns logits/argmax/final_argmax/agree/rank_of_final/top_logit (argmax-family fields compared exactly, no tolerance).
- Oracle: `test_cases._py` pure Python (math.sqrt + hand-written argmax), comparing all fields.
- Anchors: (1) h=[4,0]: normed=±2/√(4+1e-5), logits=[e,−e,0], the two layers' argmax differ (agree=[False,True]), rank_of_final=[3,1]; (2) apply_norm=False raw-readout contrast; (3) all-tied logits → argmax=0 (smallest index pinned); (4) rank==1 ⇔ agree (self-consistency invariant); (5) b shifts the linear combination directly; (6) constant hidden (var=0) readout determined by eps and b alone.
- Invalid inputs: shapes, empty input, non-bool apply_norm, eps≤0 → ValueError.

## Mutation log (9, all caught)

no-layernorm (the classic logit-lens transcription bug), var-unbiased (divide by D−1), eps-drop, no-affine, tie-largest, rank-inclusive (≥ includes itself), agree-inverted, final-from-first, unembed-transpose.

## Edge cases and pitfalls

- Anchor math must include eps: ±2/√(4+1e-5) ≈ ±0.99999875; 1e-12-level asserts need the analytic form.
- torch's tie behavior for argmax is unspecified; the problem pins "smallest index" with a hand-written implementation.
- Red lines: no real-model forward, no probe-parameter training (tuned lens), no soft-label CE/KL (that lives in 653).
