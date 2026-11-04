"""DeFT — starter template.

Implement `solve(mode, **kw)` reproducing the pure-function core of HuggingFace PEFT's
deft tuner (peft/tuners/deft/layer.py; decoupled fine-tuning):

- resolve_r: effective rank capped at out_features (r <= 0 raises).
- resolve_scaling: alpha / r, or 1.0 without alpha.
- project: (Q_P, right) with P_proj = Q_P @ right.T — "relu": Q_P = P, right =
  relu(P) (non-orthogonal); "qr": Q_P = qr(P), right = Q_P (orthogonal). All in fp32.
- merge_factor: M = right.T @ W (r x in), W logical (out, in).
- delta_from_factor: delta = Q_P @ (R * scaling - M); para => delta = -Q_P @ M.
- init_R: R = right.T @ W / scaling so the initial delta is exactly zero.
- branch: result = base_product - (base_product @ right) @ Q_P^T
  + F.linear(x, R * scaling) @ Q_P^T (full mode only); equals base_product + x @ delta^T.

See description.md for the full contract.
"""

MODES = ["resolve_r", "resolve_scaling", "project", "merge_factor",
         "delta_from_factor", "init_R", "branch"]


def solve(mode, **kw):
    raise NotImplementedError
