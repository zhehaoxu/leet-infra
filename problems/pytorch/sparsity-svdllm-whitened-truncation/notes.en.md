# Solution Notes

## Approach

The per-Linear compression pipeline of the official SVDLLM.py, taking two deterministic segments:

1. **whitening** (L187-284): whitened-domain SVD + parameter-budget truncation + √Σ factor split (v-side L⁻¹ compensation);
2. **local_update / fasterprune** (L442-498): lstsq closed-form update + append factors + relative errors.

Outside the judging scope: calibration H accumulation (the Σinpᵀinp hook), Cholesky producing L (profiling stage), and the non-PSD fix (the eigenvalue branch lives on the profiling side; only the inv-failure `+1e-6·I` defense is kept here).

Boundary with existing problems:
- **771 LoSparse**: W's own spectrum + √diag split + residual structured pruning; no whitening, no L⁻¹;
- **520 SparseGPT**: column-wise OB pruning with error propagation, not global low-rank;
- **770 SliceGPT**: orthogonal rotation with rank-preserving slicing; SVD-LLM is rank-reducing approximation.

## Key implementation points

- **The k upper bound is dead code**: with `ratio ≤ 1`, `k = int(MN/(M+N)) ≤ min(M,N)/2 < min(M,N)` always holds (MN/(M+N) is half the harmonic mean of the two dims), so only `k < 1` is checked — the mirror image of 771's "positive ratio through ceil is always ≥ 1".
- **The √Σ split's math**: `svd_u @ svd_v = U_k Σ_k V_k L⁻¹`; `svd_u @ (svd_v @ L) = U_k Σ_k V_k`, exactly the rank-k optimal approximation of `W@L` (Eckart–Young) — the test asserts relative error < 0.5 (at k=3 with slow spectral decay an absolute error of 2.28 is normal).
- **The official updated_output convention**: `x @ truc_vᵀ @ Σ @ updated_uᵀ`, which is *not* algebraically equal to the `append_u @ append_v` composition (√Σ sandwich: Σ is diagonal but √Σ@A@√Σ ≠ Σ@A) — the judging replicates both faithfully, with `updated_error` defined via the append composition's output.
- **L fixture**: the Cholesky of `H = raw@rawᵀ + 0.5·I` — guaranteed PD and invertible; the singular case zeroes one row of L to trigger the `+1e-6·I` defense (inv-failure branch), and the result must be finite and agree with the oracle's `np.linalg.inv` defense.
- **The direct_update branch**: `scaling_diag_matrix=None` and passing L is an error; officially `direct_update=True` runs `svd(W)` with no L⁻¹ on the v side.

## Test design (8 cases)

- Full oracle comparison on both whitened/direct paths (k, factors, reconstruction, updated_uᵀ, append factors, both errors);
- L⁻¹ observable: svd_v ≠ naive √Σ@Vᵀ (the criterion for whitening-loss mutants) plus `svd_u@(svd_v@L) ≈ W@L` with bounded relative error;
- Closed-form compensation never worsens: `updated_error ≤ error` (lstsq least-squares property); error monotonically non-increasing in k;
- Singular-L defense: finite results, oracle-consistent;
- k formula: four-point monotonicity in ratio + ratio=1.0 legal + too-small ratio raises `k (0) 必须 >= 1`;
- Small-norm y (×0.005, with x scaled in lockstep to preserve compatibility): catches `max(norm,1)`-style denominator mutants;
- Purity + validation (shapes / integer tensors / direct+L combination / ratio bounds and bool).

## Mutation closure

All 39 mutations caught (no equivalent exclusions). Coverage: four k-formula mutants + lower-bound-check removal + message; whitening loss / transpose / direct flip; tail-reversal on all three truncation sides + k±1; three √Σ mutants (Σ on the u side, Σ on the v side, sqrt dropped); L⁻¹ dropped / replaced by L / defense-ε mutant / defense removed; four closed-form-update mutants (Σ dropped, Vᵀ not transposed, lstsq args swapped, updated_uᵀ replaced by stale truc_u); sqrt mutants on append_v/new_w; error denominator / numerator / updated-using-stale-reconstruction; seven validation mutants.

## Engineering remarks

- Test dimensions must make `k = int(MN·ratio/(M+N)) ≥ 1`: 12×8@0.2 gives k=0 and every case errors out; enlarged to 40×24 (k=3 at 0.2).
- Scaling y alone breaks x/y compatibility and explodes the lstsq error — scale x and y in lockstep (`x = x0·s`, `y = (x0@Wᵀ + noise)·s`).
- Do not hardcode dimension strings in f-string message assertions (after resizing, the stale `(8, 8)` strings caused two rounds of false args_validation failures).
