"""OSF — starter template.

Implement `solve(mode, **kw)` reproducing the pure-function core of HuggingFace PEFT's
osf tuner (peft/tuners/osf/utils.py; continual learning with adaptive SVD):

- decompose: full SVD of W (computed in float32), split into frozen high part
  (first k = min(top_k, min_dim) components) and trainable low part (the rest).
- reconstruct: U_high diag(S_high) V_high + U_low diag(S_low) V_low — exact inverse.
- resolve_rank: effective_rank int -> as-is; float in (0, 1] -> int(fraction *
  min_dim); None -> min_dim // 2. effective_rank is the PRESERVED rank.
- project_u: dU <- dU - U_high @ (U_high^T @ dU)  (orthogonal to U_high columns).
- project_v: dV <- dV - dV @ (V_high^T @ V_high)  (orthogonal to V_high rows).

See description.md for the full contract.
"""

MODES = ["decompose", "reconstruct", "resolve_rank", "project_u", "project_v"]


def solve(mode, **kw):
    raise NotImplementedError
