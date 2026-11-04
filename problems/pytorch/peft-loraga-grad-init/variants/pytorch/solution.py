"""PEFT: LoRA-GA gradient-approximation initialization (official Outsider565/LoRA-GA, arXiv:2407.05000).

Judging boundary is float64; the SVD pipeline follows the official float32 flow.
"""
import math
import random

import torch


def grad_svd(grad: torch.Tensor, r: int, niter: int = 4, seed: int = 42):
    """Official layer.py: svd_lowrank on the accumulated gradient (out, in), float32.

    q = min(4*r, min(grad.shape)); returns U (out, q), S (q,), Vt = V.T (q, in).
    """
    if r <= 0:
        raise ValueError("r must be positive")
    g = grad.to(torch.float32)
    torch.manual_seed(seed)
    U, S, V = torch.svd_lowrank(g, q=min(4 * r, min(g.shape)), niter=niter)
    return U.double(), S.double(), V.T.double()


def pick_direction(U: torch.Tensor, Vt: torch.Tensor, r: int,
                   direction: str, seed: int = 0):
    """Official direction assignment over the top-2r singular-vector bank.

    ArBr:  B = U[:, 0:2r:2] (even cols), A = Vt[1:2r:2, :] (odd rows) — interleaved
    A2rBr: B = U[:, :r],    A = Vt[r:2r, :]
    ArB2r: B = U[:, r:2r],  A = Vt[:r, :]
    random: python random.sample(range(2r), 2r); first half indexes A rows,
            second half indexes B cols (official draws indexes_A first)
    """
    if direction == "ArBr":
        B = U[:, 0:2 * r:2]
        A = Vt[1:2 * r:2, :]
    elif direction == "A2rBr":
        B = U[:, :r]
        A = Vt[r:2 * r, :]
    elif direction == "ArB2r":
        B = U[:, r:2 * r]
        A = Vt[:r, :]
    elif direction == "random":
        random.seed(seed)
        lst = random.sample(range(2 * r), 2 * r)
        indexes_A = lst[0:r]
        indexes_B = lst[r:2 * r]
        B = U[:, indexes_B]
        A = Vt[indexes_A, :]
    else:
        raise ValueError(f"unknown direction {direction!r}")
    return B.contiguous(), A.contiguous()


def apply_scale(B: torch.Tensor, A: torch.Tensor, scaling: float, scale: str,
                stable_gamma: float = 1.0, weight_singulars=None):
    """Official scale modes over the picked factors (returns NEW tensors).

    gd:     A /= eta, B /= eta
    unit:   untouched (A, B orthogonal)
    stable: B, A *= m**0.25 / sqrt(gamma) with m = out_features = B.size(0)
    weightS: S(w) /= eta; avg_s = sqrt(S[:r]).mean(); B, A *= avg_s
             (weight_singulars are the SVD singular values of the base weight,
              provided by the caller; the eta division happens here)
    """
    B = B.clone()
    A = A.clone()
    if scale == "gd":
        A /= scaling
        B /= scaling
    elif scale == "unit":
        pass
    elif scale == "stable":
        m = B.size(0)
        B *= m ** 0.25 / stable_gamma ** 0.5
        A *= m ** 0.25 / stable_gamma ** 0.5
    elif scale == "weightS":
        S = weight_singulars.clone() / scaling
        avg_s = torch.sqrt(S[:A.size(0)]).mean()
        B *= avg_s
        A *= avg_s
    else:
        raise ValueError(f"unknown scale {scale!r}")
    return B, A


def apply_offset(B: torch.Tensor, A: torch.Tensor, scaling: float,
                 weight: torch.Tensor, norm_clip: bool = False):
    """Official absorption: offset = B@A * eta; weight -= offset.

    norm_clip: ratio = max|weight| / max|offset|; if ratio < 1, scale
    offset *= ratio and A, B *= sqrt(ratio) — the pair (offset, A, B) keeps
    B@A*eta == offset while shrinking the absorbed update.
    Returns (new_weight, a_w, b_w).
    """
    B = B.clone()
    A = A.clone()
    offset = B @ A * scaling
    if norm_clip:
        ratio = torch.max(torch.abs(weight)) / torch.max(torch.abs(offset))
        if ratio < 1:
            offset *= ratio
            A *= ratio ** 0.5
            B *= ratio ** 0.5
    new_weight = weight - offset
    return new_weight, A.contiguous(), B.contiguous()


def solve(mode=None, **kw):
    if mode == "grad_svd":
        return grad_svd(kw["grad"], kw["r"], niter=kw.get("niter", 4), seed=kw.get("seed", 42))
    if mode == "direction":
        return pick_direction(kw["U"], kw["Vt"], kw["r"], kw["direction"], seed=kw.get("seed", 0))
    if mode == "scale":
        return apply_scale(kw["B"], kw["A"], kw["scaling"], kw["scale"],
                           stable_gamma=kw.get("stable_gamma", 1.0),
                           weight_singulars=kw.get("weight_singulars"))
    if mode == "offset_apply":
        return apply_offset(kw["B"], kw["A"], kw["scaling"], kw["weight"],
                            norm_clip=kw.get("norm_clip", False))
    raise ValueError(f"unknown mode {mode!r}")
