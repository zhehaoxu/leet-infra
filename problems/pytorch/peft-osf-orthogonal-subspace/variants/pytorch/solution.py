"""OSF — HuggingFace PEFT osf tuner (orthogonal subspace fine-tuning).

Reproduces the pure-function core of peft/src/peft/tuners/osf (utils.py; paper:
continual full fine-tuning with adaptive SVD, constraining updates to be orthogonal
to previously important directions).

- decompose_weight_matrix(W, top_k): full SVD of W (float32 compute), split into
  frozen high-rank part (first k = min(top_k, min_dim) singular components) and
  trainable low-rank part (the rest). rank_high = k.
- reconstruct_weight_matrix: W ≈ U_high @ diag(S_high) @ V_high + U_low @ diag(S_low)
  @ V_low — exact inverse of the decomposition.
- Rank resolution: effective_rank int -> preserved rank; float in (0, 1] ->
  int(fraction * min_dim); None -> 50% of min_dim. NOTE: effective_rank is the
  PRESERVED (frozen) rank; trainable = min_dim - preserved.
- Gradient projection (single-process domain):
  dU <- dU - U_high @ (U_high^T @ dU)   (orthogonalize against U_high's column space)
  dV <- dV - dV @ (V_high^T @ V_high)   (orthogonalize against V_high's row space)

Distributed all-reduce paths are out of the judging domain.
"""

import torch
from torch import nn


def solve(mode, **kw):
    if mode == "decompose":
        weight = kw["weight"]
        top_k = kw["top_k"]
        orig_dtype = weight.dtype
        W = weight.to(torch.float32)
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        k = min(top_k, S.shape[0])
        return {
            "U_high": U[:, :k].contiguous().to(dtype=orig_dtype),
            "S_high": S[:k].contiguous().to(dtype=orig_dtype),
            "V_high": Vt[:k, :].contiguous().to(dtype=orig_dtype),
            "U_low": U[:, k:].contiguous().to(dtype=orig_dtype),
            "S_low": S[k:].contiguous().to(dtype=orig_dtype),
            "V_low": Vt[k:, :].contiguous().to(dtype=orig_dtype),
            "rank_high": k,
        }

    if mode == "reconstruct":
        svd = kw["svd"]
        high = svd["U_high"] @ torch.diag(svd["S_high"]) @ svd["V_high"]
        low = svd["U_low"] @ torch.diag(svd["S_low"]) @ svd["V_low"]
        return high + low

    if mode == "resolve_rank":
        effective_rank = kw["effective_rank"]
        min_dim = kw["min_dim"]
        if effective_rank is None:
            return min_dim // 2
        if isinstance(effective_rank, float):
            if not (0.0 < effective_rank <= 1.0):
                raise ValueError(f"fractional effective_rank must be in (0, 1], got {effective_rank}")
            return int(effective_rank * min_dim)
        return int(effective_rank)

    if mode == "project_u":
        # dU <- dU - U_high @ (U_high^T @ dU)
        dU, U_high = kw["grad"], kw["U_high"]
        proj_coeff = torch.mm(U_high.transpose(0, 1), dU)
        return dU - U_high @ proj_coeff

    if mode == "project_v":
        # dV <- dV - dV @ (V_high^T @ V_high)
        dV, V_high = kw["grad"], kw["V_high"]
        G = torch.mm(V_high.transpose(0, 1), V_high)
        return dV - torch.mm(dV, G)

    raise ValueError(f"unknown mode: {mode}")


CHECKS = [
    "check_decompose",
    "check_reconstruct",
    "check_resolve_rank",
    "check_project_u",
    "check_project_v",
]


def check(solve):
    for name in CHECKS:
        globals()[name](solve)
