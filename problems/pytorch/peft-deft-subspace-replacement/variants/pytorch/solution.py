"""DeFT — HuggingFace PEFT deft tuner (decoupled fine-tuning).

Reproduces the pure-function core of peft/src/peft/tuners/deft/layer.py. DeFT replaces
a rank-r subspace of the weight: adapted = (I - P_proj) @ W + Q_P @ R, so

    delta = Q_P @ (R - right.T @ W)          (full DEFT)
    delta = -Q_P @ (right.T @ W)             (PaRa / para=True: pure subspace removal)

where (Q_P, right) come from the projection method:

- "relu": Q_P = P, right = relu(P)  -> P_proj = P @ relu(P).T (non-orthogonal)
- "qr":   Q_P = qr(P), right = Q_P  -> P_proj = Q_P @ Q_P.T  (orthogonal)

All projection math runs in float32. Shapes: P (out, r), R (r, in). The effective
rank is capped at out_features; scaling = alpha / r (1.0 without alpha). Identity
init: R_init = right.T @ W / scaling so the initial delta is exactly zero. The
forward computes the update on activations without materializing W + delta:

    x @ delta.T = [(x @ R.T) - (x @ W.T) @ right] @ Q_P.T

i.e. result -= (base_product @ right) @ Q_P.T; result += F.linear(x, R * scaling) @ Q_P.T.
Merge caches only the r x in factor M = right.T @ W (unmerge recomputes the exact
delta from it).
"""

import torch
import torch.nn.functional as F


def solve(mode, **kw):
    if mode == "resolve_r":
        r, out_features = kw["r"], kw["out_features"]
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        return min(r, out_features)

    if mode == "resolve_scaling":
        alpha, r = kw.get("alpha"), kw["r"]
        return (alpha / r) if alpha else 1.0

    if mode == "project":
        # float32 for numerical stability; (Q_P, right) with P_proj = Q_P @ right.T
        P = kw["P"].to(torch.float32)
        method = kw["method"]
        if method == "relu":
            return {"Q_P": P, "right": torch.relu(P)}
        if method == "qr":
            Q_P, _ = torch.linalg.qr(P)
            return {"Q_P": Q_P, "right": Q_P}
        raise ValueError(f"Unknown decomposition_method '{method}'.")

    if mode == "merge_factor":
        # M = right.T @ W  (r x in), W logical (out, in) in float32
        P, W, method = kw["P"], kw["W"].to(torch.float32), kw["method"]
        right = solve(mode="project", P=P, method=method)["right"]
        return right.transpose(0, 1) @ W

    if mode == "delta_from_factor":
        # delta = Q_P @ (R * scaling - M); para => delta = -Q_P @ M
        P, factor = kw["P"], kw["factor"]
        method, para = kw["method"], kw.get("para", False)
        Q_P = solve(mode="project", P=P, method=method)["Q_P"]
        if para:
            return -(Q_P @ factor)
        R = kw["R"].to(torch.float32) * kw["scaling"]
        return Q_P @ (R - factor)

    if mode == "init_R":
        # Identity init: R = right.T @ W / scaling so that (scaling * R) == right.T @ W
        # and the initial delta is exactly zero.
        P, W, method = kw["P"], kw["W"].to(torch.float32), kw["method"]
        scaling = kw["scaling"]
        right = solve(mode="project", P=P, method=method)["right"]
        return right.transpose(0, 1) @ W / scaling

    if mode == "branch":
        # Activation-domain update: result -= (base_product @ right) @ Q_P^T;
        # result += F.linear(x, R * scaling) @ Q_P^T  (full mode only).
        x, base_product = kw["x"].to(torch.float32), kw["base_product"].to(torch.float32)
        P, R = kw["P"], kw["R"]
        method, para = kw["method"], kw.get("para", False)
        scaling = kw["scaling"]
        proj = solve(mode="project", P=P, method=method)
        Q_P, right = proj["Q_P"], proj["right"]
        result = base_product - (base_product @ right) @ Q_P.transpose(0, 1)
        if not para:
            R = R.to(torch.float32) * scaling
            result = result + F.linear(x, R) @ Q_P.transpose(0, 1)
        return result

    raise ValueError(f"unknown mode: {mode}")


CHECKS = [
    "check_resolve",
    "check_project",
    "check_delta_identity",
    "check_branch",
]


def check(solve):
    for name in CHECKS:
        globals()[name](solve)
