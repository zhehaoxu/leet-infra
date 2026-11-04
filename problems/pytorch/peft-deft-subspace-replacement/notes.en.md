# Solution notes — DeFT

## Official account

- HF PEFT `peft/tuners/layer.py` (the DeFT layer) + `decomposition.py` (relu/qr
  projection methods) + docs; decoupled fine-tuning via subspace replacement
  (derived from PaRa).
- Judging domain: rank resolution, projection decomposition, merge/unmerge
  factors, the delta identity, activation-domain branch forward. Training loops
  and data parallelism are out of scope.

## Pitfalls

- **Unpacking a dict yields key-name strings**: `Q_P, right = solve(mode="project",
  ...)` destructures the returned dict into two key-name strings, so the following
  matmuls raise TypeError — use `out["Q_P"]` key access instead.
- **branch returns the full output**: not a pure delta — it is
  `base_product - (base_product @ right) @ Q_P.T` (removal term) `+
  F.linear(x, R * scaling) @ Q_P.T` (injection term); the identity is
  `out == base_product + x @ delta.T`, and asserting `out == x @ delta.T` fails
  everywhere.
- **The correct identity-init assertion**: `R_init = right.T @ W / scaling` makes
  the initial delta exactly zero, hence `out0 == base_product` (not "delta is any
  zero tensor"); in para mode R is ignored and only the removal term remains
  (delta nonzero).
- **fp32 pipeline tolerance**: projection/merge/branch all force
  `to(torch.float32)`, so under a float64 test the identities differ by ~1e-6 —
  tolerance 1e-5 (same root cause as OSF/CorDA); dtype-matched comparisons need
  `to(torch.float32)`.
- **A placeholder anchor value is a live FAIL truth**: anchors must be confirmed
  by running the solution first, never pre-filled with guesses (otherwise a
  always-false assertion masquerades as passing).
- **The relu projection is not orthogonal**: `P_proj = P @ relu(P).T` is not an
  idempotent orthogonal projector (only qr is) — assert `right == relu(P)` for the
  relu branch, never orthogonality/idempotence.

## Mutant list (14)

resolve: min→max (clamp direction flipped), alpha/r→alpha*r; project:
relu→sigmoid (right factor), qr(P)→qr(P.T) (wrong orthonormal basis);
merge_factor: W→W.T (shape/semantics), right.T@W→right@W; delta: R−factor→
factor−R (sign), para minus dropped, R*scaling→R+scaling; init_R: /scaling→
*scaling; branch: removal term dropped, injection +→−, Q_P.T→Q_P (transpose
misplaced), Q_P/right swapped in the removal term (misplaced correction). 14/14
CAUGHT.
