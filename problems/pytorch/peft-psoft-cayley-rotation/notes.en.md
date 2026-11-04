# Solution notes — PSOFT

## Official account

- HF PEFT `peft/tuners/psoft/layer.py` (OrthLayer + PsoftLayer); the judging domain
  covers the pure-function parts.
- **How the judging-domain concern was resolved**: the earlier worry was "SVD oracle
  independence over data" — in fact the SVD input W is given directly by the test
  cases, and full-mode `torch.linalg.svd` is fully deterministic; combined with
  orthogonality/identity assertions (R·Rᵀ=I, det=+1, R=I ⇒ zero delta,
  branch ≡ x@ΔWᵀ) the mutants are all caught. PSOFT is judgeable after all.

## Pitfalls

- **Placeholder anchors with `< 0.0`**: always-false assertions (abs >= 0); the first
  version failed at two spots — anchors must be real values.
- **left=False semantics**: `solve(A, B, left=False)` solves X·A = B (X = B·A⁻¹),
  not the usual A·X = B. The Cayley (I+Q)(I−Q)⁻¹ uses left=False; left=True is
  numerically equivalent (Q's rational functions commute) — a removed true equivalence.
- **Neumann series tail**: terms 1..t−2 carry coefficient 2.0, the final term
  (Q^(t−1)) carries 1.0 — `R.add_(Q_power)` without alpha. The nested terms>1/2/3
  branches make miscounting easy.
- **Projection cross-check**: the solution's Q comes from weight=w·100; when the test
  hand-computes from Q=skew(w) it must multiply by 100 — linearity
  skew(c·w) = c·skew(w) holds but is easy to miss.
- **Three true equivalences (removed)**: Cayley solve direction (commutativity),
  projection `>` vs `>=` (the float boundary cannot be constructed stably), and the
  +1e-12 guard (relative difference < 1.2e-12 under the norm > 0.9 constraint).
- SVD sign ambiguity is not an issue (deterministic for a fixed input), but anchors
  must be taken from computed values, not hand-guessed.

## Mutant list (14)

skew: Q−Qᵀ→Q+Qᵀ, upper→lower triangle indices; orth: solve direction sign flip,
Neumann first-term coefficient, tail coefficient, Q² term degenerated; magnify:
row/column scaling vectors swapped; svd: head→tail truncation; init: psoft B loses
scaling, pissa loses sqrt; delta: R−I→I−R (sign), scaling ×→÷; branch: (xr−xa) sign,
Rᵀ→R. 14/14 CAUGHT.
