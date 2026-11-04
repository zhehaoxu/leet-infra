# Notes

- Source: ShadowPEFT (arXiv:2604.19254), official ShadowLLM/shadow-peft `src/shadow_peft/modules.py` (commit 65a43a7, merged into HF PEFT).
- Judging domain: the pure-function cores of ShadowInjectionModel and ShadowUpdateModel + parameter counts + init RNG replay + the gate algebra identity; the shadow backbone and KV cache are out of scope (dropout at eval semantics, p=0).
- Modes (8): inject / inject_zero_init / inject_sequence / update / gate / gated_identity / param_count / init_injection.
- Weight conventions: gate/transform weights follow nn.Linear (out, in) (F.linear); injection's W_down (d, k) and W_up (k, d) are used directly per the einsum subscripts (matching the official Parameter shapes) — two different conventions, careful in manual anchor replays.
- Anchors in float64, tolerance 1e-9; downs is one single normal_(std=0.02) block draw of the whole (L, d, k) tensor, not per-layer draws.
- Paper deviations: T (transform) has no sigmoid; sigmoid lives only in gate G; dropout only mid-transform; LN shared, eps=1e-5.
- mut 17: difference sign / alpha direction / up subscript / down transpose / zero-init flip / sequence skip / LN eps / gate relu / gate SiLU removal / transform SiLU→ReLU / gated formula / gate flip / identity flip / two parameter-count sites / init std (solve layer) / nonzero ups — all CAUGHT. A solve-layer explicit default shadows function-signature-default mutations (same lesson as Arrow).
