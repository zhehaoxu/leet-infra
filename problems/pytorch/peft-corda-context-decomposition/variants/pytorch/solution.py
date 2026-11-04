"""PEFT: CorDA context-oriented decomposition adaptation (official iboing/CorDA,
cordalib/decomposition.py, arXiv:2406.05223). Judging domain: pure-tensor rebuild of
the cov_aware decomposition account.

Official account (decompose_to_adapter, cov_aware branch):
  1. damped inverse of the covariance matrix: fix = cov + I * mean(diag(cov)) * damp,
     damp starts at 0.01 and doubles while ||fix @ inv(fix) - I|| >= 0.05.
  2. w = W @ fix_covariance;  U, S, V = linalg.svd(w);  V <- (V.T @ cov_inv).T.
     Context identity: U diag(S) V^T == W exactly (the covariance cancels).
  3. Component selection: first_eigen=False (KPA) keeps the LAST r components;
     first_eigen=True (IPA, following PiSSA) keeps the FIRST r.
  4. weight_residual = W - U diag(S) V^T (frozen bypass).
  5. sigma_fuse: "UV" -> A=U*sqrt(S), B=V^T*sqrt(S) (column scaling on both);
     "U" -> A=U*S, B=V^T;  "V" -> A=U, B=V^T*S. All three satisfy A @ B = U diag(S) V^T.
  6. forward: y = BLinear(x) -> ALinear(y) + x @ residual^T; at init this equals x @ W^T.
"""
import math

import torch


def damped_cov_inverse(cov: torch.Tensor, damp: float = 0.01, tol: float = 0.05):
    """Official damp loop: double damp until the damped covariance inverts cleanly."""
    while True:
        compensate = torch.diag(
            torch.ones(cov.size(0), dtype=cov.dtype) * torch.mean(torch.diag(cov)) * damp
        )
        fix = cov + compensate
        cov_inv = torch.linalg.inv(fix)
        inv_error = torch.dist(fix @ cov_inv, torch.eye(cov.size(0), dtype=cov.dtype))
        if inv_error < tol:
            return fix, cov_inv, damp
        damp *= 2


def corda_decompose(weight: torch.Tensor, cov: torch.Tensor, r: int,
                    first_eigen: bool = False):
    """Official cov_aware decomposition. Returns (U_r, S_r, V_r, residual, fix, cov_inv, damp)."""
    if r <= 0:
        raise ValueError("r must be positive")
    m, n = weight.shape
    rank = min(m, n)
    if r > rank:
        raise ValueError("r exceeds the matrix rank")
    fix, cov_inv, damp = damped_cov_inverse(cov.float())
    w = weight.float() @ fix
    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    V = Vh.transpose(0, 1)
    V = (V.t() @ cov_inv).transpose(0, 1)
    if torch.isnan(S).any() or torch.isinf(S).any():
        raise ValueError("nan or inf in S")
    if first_eigen:
        U, S, V = U[:, :r], S[:r], V[:, :r]
    else:
        U, S, V = U[:, -r:], S[-r:], V[:, -r:]
    residual = weight.float() - U @ torch.diag(S) @ V.t()
    if torch.isnan(residual).any() or torch.isinf(residual).any():
        raise ValueError("nan or inf in weight_residual")
    return U, S, V, residual, fix, cov_inv, damp


def sigma_fuse(U: torch.Tensor, S: torch.Tensor, V: torch.Tensor,
               mode: str = "UV"):
    """Official sigma_fuse: split U diag(S) V^T into (A, B) factor weights."""
    if mode == "UV":
        return U.mul(S.sqrt()).contiguous(), V.t().mul(S.sqrt().view(-1, 1)).contiguous()
    if mode == "U":
        return U.mul(S).contiguous(), V.t().contiguous()
    if mode == "V":
        return U.contiguous(), V.t().mul(S.view(-1, 1)).contiguous()
    raise ValueError(f"unknown sigma_fuse {mode!r}")


def adapter_forward(x: torch.Tensor, a_w: torch.Tensor, b_w: torch.Tensor,
                    residual: torch.Tensor) -> torch.Tensor:
    """Official CorDA_adapter.forward: BLinear -> ALinear + frozen residual bypass."""
    return torch.nn.functional.linear(
        torch.nn.functional.linear(x, b_w), a_w
    ) + torch.nn.functional.linear(x, residual)


def solve(mode=None, **kw):
    if mode == "damped_inv":
        return damped_cov_inverse(kw["cov"], kw.get("damp", 0.01), kw.get("tol", 0.05))
    if mode == "decompose":
        return corda_decompose(kw["weight"], kw["cov"], kw["r"],
                               kw.get("first_eigen", False))
    if mode == "fuse":
        return sigma_fuse(kw["u"], kw["s"], kw["v"], kw.get("fuse", "UV"))
    if mode == "forward":
        return adapter_forward(kw["x"], kw["a_w"], kw["b_w"], kw["residual"])
    raise ValueError(f"unknown mode {mode!r}")
