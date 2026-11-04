# Arrow Implementation Notes

## Source and Structure

- Official PEFT `src/peft/tuners/lora/arrow.py` (`ArrowLoraLinearLayer`, PR #2644, same PR as GenKnowSub); paper arXiv:2405.11157 (NeurIPS 2024, Mila).
- Prototype = the top right singular vector of dW=B@A; power-iteration operator `v -> A^T (B^T B) (A v)` — never materializing dW (out×in) or M=A^T(B^T B)A (in×in); fp32 iteration; `rng_seed` builds a private Generator deciding v₀.
- Routing: `sim = torch.abs(tok @ protos.T)` (sign-agnostic); topk -> -inf for non-selected -> `softmax(full_score / temperature)`; three einsums (z/y/delta) plus per-adapter scaling; dropout applies after delta_flat (identity in the judging domain).
- `on_adapter_change` invalidates `_protos_ready` for lazy rebuilds; `check_loaded_lora_compatibility_arrow` validates rank/shapes/module sets.

## Judging Conventions and Quirks

- **Power-iteration equivalence classes**: initial-value perturbations, the ± direction, and iteration counts beyond convergence are all genuinely equivalent (15 vs 14 diff=0, converged within fp32 precision); catchable are UNDER-iteration (iters=2 vs 15 differ by 0.26) and operator/matrix-order mutations.
- The solve dispatch layer hard-coding `kw.get("iters", 15)` SHADOWS the function signature default — mutating the signature default is then ineffective; the mutation point must live in the solve layer (the flip side of the 809 lesson).
- **The fp32 -> LoRA-dtype cast-back** is the official build_prototypes behavior (`proto32.to(dtype=A.dtype)`); in the fp64 judging domain the routing runs fully fp64 and the dtype assertion follows A, not fp32.
- The abs in the similarity is the official sign-agnostic design; removing it is a catchable mutation (tests include tokens with negative similarities).
- The -inf fill makes non-selected softmax coefficients **exactly 0** (not small numbers); asserted with `torch.equal(coeff[~mask], 0)`.
- Equal-score ties (a zero token has similarity 0 to every prototype) are resolved by topk index order — used by the tests to verify the even 0.5/0.5 split.
- The three compatibility ValueErrors (rank/shape/modules) must NOT back each other up: the rank case must use identical shapes, otherwise a rank mutation survives through the shape check.
- Anchor tolerances: singular_vec 1e-7 (fp32 iteration precision), routing/combination 1e-9 (fp64 linear algebra).
