# Solution Notes: SliceGPT Orthogonal Rotate-and-Slice Step

## Source Localization

- Source: microsoft/TransformerCompression (repo layout src/slicegpt/; the old slicegpt/ path 404s), `rotate.py`, 528 lines. A completely different family from the existing 16 sparsity problems (rotation-based rank reduction vs zeroing/reconstruction/scheduling) — no overlap.
- `pca_calc` (L501-528): `H_batch = Σ_batch x.mT @ x` (per-batch mask zeroing first) → damp = 0.01·mean(diag) added to the diagonal only → eigh ascending → descending argsort reordering both values and column vectors.
- `rotate_and_slice_sequential` (L140-251): four rotations (attn inputs right-multiplied by Q, attn output/MLP output left-multiplied by Q.T, MLP input right-multiplied by Q) + four slices (three column, two row) + the two shortcut chains; `slice_attention_inputs` slices the shortcut's **rows** (L32), and `get_mlp_input_dimension ≡ get_attention_output_dimension` (ForwardSlicingScheduler L194-196).
- Dtype discipline: weight rotation in float64 then cast back; the shortcut chain casts `Q.to(dtype)` (float32) before matmul — at float32 output granularity the float64/float32 matmul difference is < 2 ULP, indistinguishable by judging tolerance, so dtype-boundary mutants are excluded from the list (the spec text still requires float64).
- The official pca masks in place (mutating inps); the judgement domain clones for purity.

## Math Notes

1. **Rotate-and-slice = rank reduction**: W' = (W Q)[:, :k] discards the low-energy coordinates in the principal-component basis; the reconstruction property W' Q[:k].T ≈ W holds approximately only when slicing the top-k components (information loss concentrated in the tail eigenvalues). Tests use the sign-agnostic properties Q^T Q = I and H q = q·diag(vals).
2. **Damping's dual role**: prevents numerical degeneracy in eigh, and **shifts the spectrum without changing eigenvectors** — so the eig_vals assertion (vals = vals_nodamp + damp) pins the damp formula while distinguishing damp-all (H + damp·11^T changes vectors) from damp-drop (no shift).
3. **A non-degenerate spectrum is a test precondition**: within a degenerate eigenspace eigh may return any basis. The uniform ripple of `_sig` after mask zeroing produced ~5e-12 near-degenerate pairs (measured); switching the mask multiplier to 0.37 gives a min gap of 1.8e-2 — **the mask pattern and data pattern must be scanned jointly**.
4. **Batch accumulation ≡ concatenation for H**: Σ_b Σ_token x xᵀ vs a single cat(batch) accumulation differ only in float summation order (~1e-13); verified with allclose 1e-7.
5. **The shortcut chain is a cross-block basis change**: attn_shortcut maps attention-output coordinates into the next sub-block's sliced coordinates; the cast-then-slice vs slice-then-cast orders are equivalent (column selection commutes with matmul) — what matters is slicing the wrong axis or not slicing.

## Implementation Notes

- Validation order: ignore_masks is a list or None → three signal groups (non-empty list, 3-D, consistent hidden dim, mask alignment + 0/1) → equal batch counts → w_qkv list non-empty with (d,d) shapes → w_o/w1/w2 shapes → o_bias → three slice dims (bool excluded, [1,d]).
- A single-element list with a (B,S,3) tensor passes the 3-D/hidden-dim checks but hits the batch-count check first — **the validation order dictates how test cases must be constructed** (the non-3-D case must use a 2-D tensor).
- PCA returns (vals_desc, vecs_desc); called three times, with q and vals both returned for property assertions.

## Testing Notes

- **Layered oracle**: H construction independently recomputed in numpy (einsum per-token outer products + damping), eigh as a shared primitive (torch and numpy LAPACK sign conventions may differ, so comparing Q matrices directly is unsafe), bookkeeping mirrored — any mutation of H propagates into the weights.
- Sign-agnostic property assertions: Q^T Q = I, H q = q·diag(vals), vals descending with adjacent gaps > 1e-3 (the non-degeneracy assertion doubles as a guardrail).
- The damping assertion: vals == vals_nodamp + damp (against the damping=False variant of _np_h).
- Mask semantics: masked run == manually-zeroed no-mask run (multiplying by the mask equals zeroing); masked vs unmasked results differ.
- Batch accumulation: two batches vs a concatenated single batch give the same spectrum/vectors.
- Reversibility at full dims: W' Q.T ≈ W (reconstruction); identical signals in all three groups → q_in == q_mid == q_out.
- Purity (the mask-zeroing path must clone), the bias-None path, and a validation battery (empty list/2-D/inconsistent hidden dim/batch counts/mask alignment and 0-1/shapes/dim bool and range).

## Mutation List (44, all CAUGHT, no equivalents)

PCA 11: damp dropped/coefficient/added-everywhere/mean→min, mask inverted/dropped/in-place, x.mT swapped, batch accumulation taking the last batch, argsort ascending, vectors unreordered. Rotation/slice 11: qkv wrong side/unsliced/row-sliced, wo wrong side/unsliced, bias unsliced/unrotated, w1 wrong side/unsliced, w2 wrong side/unsliced. Shortcut 10: attn not transposed/rows unsliced/Q_mid columns unsliced/swapped to Q_out, mlp rows unsliced/swapped to Q_in/Q_out multiply dropped/columns unsliced/trailing transpose. Validation 12: signals empty list/dim/hidden-dim consistency/batch counts/mask list/alignment/0-1, w_qkv empty/shape, w1w2 dims, o_bias, dim bool, message.

Experience: ① **design the mask zeroing and data generation jointly** — masking can degenerate the spectrum (5e-12 gaps), making eigh's in-subspace choice arbitrary and the oracle comparison flaky; scan for a gap > 1e-3 combination and assert the gap explicitly (a self-guarding guardrail); ② the layered oracle of **shared primitives (eigh) + independent construction (H)** is the judging solution for eigendecomposition problems — torch/numpy LAPACK sign conventions differ so a fully independent oracle is impossible, but the independence of H construction and the bookkeeping pipeline exposes every mechanism mutant; ③ validation order dictates case construction (a self-consistent single-element list skips the hidden-dim check and reaches the batch-count check); ④ at float32 output granularity, dtype-boundary mutants (rotating via float32 matmul) differ by < 2 ULP and are unobservable — excluded from the mutation list, constrained by the spec text.
