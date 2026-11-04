# Solution notes — OSF

## Official account

- HF PEFT `peft/tuners/osf/utils.py` (decompose_weight_matrix /
  project_gradient_to_orthogonal_space) + docs (effective_rank semantics,
  reconstruct); continual learning with adaptive SVD.
- Judging domain: SVD split, reconstruct inverse, rank resolution, single-process
  gradient projection. Distributed all-reduce is out of scope.

## Pitfalls

- **The test copy cannot be named `os.py`**: it shadows the stdlib `import os`, and
  `__import__('os')` returns the stdlib module (AttributeError: module 'os' has no
  attribute 'solve').
- **Projection orthogonality needs orthonormal factors**: with random-Gaussian
  U_high/V_high, `Uᵀ(proj) = 0` does not hold (it requires UᵀU = I) — build them via
  `torch.linalg.qr(randn)[0]` (V_high as `.T` for orthonormal rows). Root cause of
  the two failing orthogonality assertions in the first run.
- **fp32 pipeline tolerance**: the official decompose forces `to(torch.float32)` and
  casts back, so under a float64 test the reconstruction / best-approximation
  identities differ by ~1e-6 — tolerance 1e-5 (same lesson as CorDA).
- **effective_rank is the preserved rank**: trainable = min_dim − preserved, opposite
  of LoRA's r; cover the float-fraction branch, the None default of 50%, and the
  out-of-range ValueError.
- **The U/V projection formulas are asymmetric**: U left-multiplies a projector, V
  right-multiplies a Gram matrix — copying one side to the other is wrong.

## Mutant list (15)

decompose: head→tail truncation, min clamp removed, rank_high ±1, U_low boundary
shift; reconstruct: high−low, missing diag (shape error); resolve: None 0.5→0.25,
int→round, fraction boundary includes 0; project_u: return grad as-is, mm transpose
misplaced (shape error), minus→plus; project_v: return as-is, Gram shape error
(VVᵀ), minus→plus. 15/15 CAUGHT.
