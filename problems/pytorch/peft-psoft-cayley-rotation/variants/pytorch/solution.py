"""PSOFT — HuggingFace PEFT psoft tuner (orthogonal rotation between SVD factors).

Reproduces the pure-function core of peft/src/peft/tuners/psoft/layer.py:

- OrthLayer: an r x r orthogonal matrix R parametrized by size(size-1)/2 upper-triangular
  entries forming a skew-symmetric Q, then the Cayley transform R = (I+Q)(I-Q)^-1
  (via linalg.solve(..., left=False)), with an optional Cayley-Neumann series
  R = I + 2Q + 2Q^2 + 2Q^3 + ... (coefficient 2.0 on terms 1..t-2, final term 1.0)
  and an optional Frobenius-norm projection of Q (eps, scaling eps/(norm+1e-12)).
- Magnitude vectors: R = diag(vector_b) @ R @ diag(vector_a) — row scaling via
  vector_b[:, None] * R, column scaling via R * vector_a[None, :].
- SVD factor cache (full mode): W (out, in) -> Vr = U[:, :r], Sr = S[:r], Uhr = Vh[:r].
- A/B init: psoft_init  => A = Uhr, B = Vr @ diag(Sr / scaling);
            pissa_init  => A = diag(sqrt(Sr/scaling)) @ Uhr, B = Vr @ diag(sqrt(Sr/scaling)).
- Delta: dW = scaling * B (R - I) A  — zero when R = I (identity at init).
- Forward branch: xa = x @ A^T; xr = xa @ R^T; branch = (xr - xa) @ B^T * scaling.

The judging domain fixes the SVD input (given W), so torch.linalg.svd (full mode) is
deterministic; identities (R orthogonal, R=I zero delta, delta/branch consistency)
pin the rest.
"""

import torch

EPS_PSOFT = 1e-12


def solve(mode, **kw):
    if mode == "skew_symmetric":
        weight = kw["weight"]           # (size*(size-1)//2,)
        size = kw["size"]
        rows, cols = torch.triu_indices(size, size, 1)
        Q = torch.zeros((size, size), dtype=weight.dtype)
        Q = Q.index_put((rows, cols), weight)
        return Q - Q.transpose(0, 1)

    if mode == "orth_matrix":
        weight = kw["weight"]
        size = kw["size"]
        use_neumann = kw.get("use_cayley_neumann", False)
        terms = int(kw.get("num_cayley_neumann_terms", 5))
        neumann_eps = kw.get("cayley_neumann_eps")
        Q = solve(mode="skew_symmetric", weight=weight, size=size)
        if neumann_eps is not None:
            norm = torch.linalg.norm(Q, ord="fro")
            if torch.isfinite(norm) and norm > neumann_eps:
                Q = Q * (neumann_eps / (norm + EPS_PSOFT))
        eye = torch.eye(size, dtype=Q.dtype)
        if use_neumann:
            R = eye.clone()
            if terms > 1:
                R.add_(Q, alpha=2.0)
                if terms > 2:
                    Q_squared = Q @ Q
                    R.add_(Q_squared, alpha=2.0)
                    if terms > 3:
                        Q_power = Q_squared
                        for _ in range(3, terms - 1):
                            Q_power = Q_power @ Q
                            R.add_(Q_power, alpha=2.0)
                        Q_power = Q_power @ Q
                        R.add_(Q_power)
            return R
        # R = (I+Q)(I-Q)^-1  — solve(..., left=False) solves X @ (I-Q) = (I+Q)
        return torch.linalg.solve(eye - Q, eye + Q, left=False)

    if mode == "magnify":
        # Official: row scaling first (vector_b), then column scaling (vector_a).
        R, vector_b, vector_a = kw["R"], kw["vector_b"], kw["vector_a"]
        R = vector_b[:, None] * R
        R = R * vector_a[None, :]
        return R

    if mode == "svd_factors":
        W = kw["W"]                     # (out, in)
        r = kw["r"]
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        return {"Vr": U[:, :r], "Sr": S[:r], "Uhr": Vh[:r, :]}

    if mode == "init_ab":
        f = kw["factors"]
        Vr, Sr, Uhr = f["Vr"], f["Sr"], f["Uhr"]
        scaling = kw["scaling"]
        init_type = kw["init_type"]
        Sr_scaled = Sr / scaling
        if init_type == "psoft_init":
            A = Uhr
            B = Vr @ torch.diag(Sr_scaled)
        elif init_type == "pissa_init":
            s_sqrt = torch.sqrt(Sr_scaled)
            A = torch.diag(s_sqrt) @ Uhr
            B = Vr @ torch.diag(s_sqrt)
        else:
            raise ValueError(f"Unknown ab_svd_init: {init_type}")
        return {"A": A.contiguous(), "B": B.contiguous()}

    if mode == "delta":
        # dW = scaling * B (R - I) A
        B, R, A = kw["B"], kw["R"], kw["A"]
        r = kw["r"]
        scaling = kw["scaling"]
        eye = torch.eye(r, dtype=R.dtype)
        return (B @ (R - eye) @ A) * scaling

    if mode == "branch":
        # forward adapter branch: xa = x @ A^T; xr = xa @ R^T;
        # branch = (xr - xa) @ B^T * scaling
        x, A, B, R = kw["x"], kw["A"], kw["B"], kw["R"]
        scaling = kw["scaling"]
        xa = x @ A.t()
        xr = xa @ R.t()
        return (xr - xa) @ B.t() * scaling

    if mode == "r_check":
        r, in_f, out_f = kw["r"], kw["in_features"], kw["out_features"]
        if r > min(in_f, out_f):
            raise ValueError(
                f"`r` ({r}) must be less than or equal to min(in_features, out_features) "
                f"({in_f}, {out_f})")
        return True

    raise ValueError(f"unknown mode: {mode}")


CHECKS = [
    "check_skew_symmetric",
    "check_orth_matrix",
    "check_magnify",
    "check_svd_init",
    "check_delta_branch",
    "check_r_check",
]


def check(solve):
    for name in CHECKS:
        globals()[name](solve)
