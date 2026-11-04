"""PEFT: LoRA-SB sign-estimated gradient SVD initialization (official CERT-Lab/LoRA-SB, arXiv:2411.19557).

All computations in float64 at the judging boundary; the SVD pipeline internally
follows the official float32 flow. Inputs/outputs are plain torch tensors.
"""
import math

import torch


def sign_update_matrix(grad: torch.Tensor, lr: float) -> torch.Tensor:
    """LoRA-SB processed gradient: dW_hat = -lr * sign(grad), sign(0) = 0."""
    return -lr * torch.sign(grad)


def _svd_grad_core(X: torch.Tensor, rank: int, niter: int, seed: int, dummy_r: bool):
    """Official get_svd_grad flow on X (in, out), float32 + fixed seed.

    Quirk: the official code draws a dummy randn(n_cols, rank) BEFORE svd_lowrank;
    the value is never used but it advances the RNG stream that svd_lowrank's
    internal random projection consumes, so removing it changes the result.
    """
    if rank <= 0:
        raise ValueError("rank must be positive")
    Xf = X.to(torch.float32)
    torch.manual_seed(seed)
    if dummy_r:
        _ = torch.randn(Xf.size(1), rank, dtype=torch.float32)  # dead value, consumes RNG
    U, s, V = torch.svd_lowrank(Xf, q=rank, niter=niter)
    enc = U @ torch.diag(s)   # (in, r)  -> A_w = enc.T
    dec = V.T                 # (r, out) -> B_w = dec.T
    return enc.double(), dec.double(), s.double()


def svd_grad(X: torch.Tensor, rank: int, niter: int = 10, seed: int = 42):
    """Official flow including the RNG-consuming dummy R."""
    return _svd_grad_core(X, rank, niter, seed, dummy_r=True)


def svd_grad_nodummy(X: torch.Tensor, rank: int, niter: int = 10, seed: int = 42):
    """Control path without the dummy draw (NOT the official flow)."""
    return _svd_grad_core(X, rank, niter, seed, dummy_r=False)


def _kaiming_uniform_(mat: torch.Tensor) -> torch.Tensor:
    """kaiming_uniform_ with a=sqrt(5): uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)).

    Mirrors torch.nn.init.kaiming_uniform_ (which exposes no generator arg);
    consumes the global RNG stream exactly like the official code.
    """
    fan_in = mat.size(1)
    bound = math.sqrt(6.0 / ((1.0 + 5.0) * fan_in))
    mat.uniform_(-bound, bound)
    return mat


def init_ab(dec: torch.Tensor, enc: torch.Tensor, half_init_dec: bool = False,
            random_init: bool = False, seed: int = 0):
    """Build LoRA A/B weights from the SVD split, with official init flags.

    Official order inside find_and_initialize(_grad):
      1) half_init_dec: re-init the LOWER half rows of dec (r, out), BEFORE transposing
      2) random_init: re-init enc fully, then dec fully (both kaiming a=sqrt(5))
      3) B_w = dec.T, A_w = enc.T
    """
    if seed >= 0:
        torch.manual_seed(seed)
    dec = dec.clone()
    enc = enc.clone()
    rows = dec.size(0)
    lower = math.ceil(rows / 2)
    if half_init_dec:
        _kaiming_uniform_(dec[lower:, :])
    if random_init:
        _kaiming_uniform_(enc)
        _kaiming_uniform_(dec)
    b_w = dec.T   # (out, r)
    a_w = enc.T   # (r, in)
    return a_w, b_w


def latent_forward(x: torch.Tensor, a_w: torch.Tensor, m: torch.Tensor,
                   b_w: torch.Tensor, scaling: float) -> torch.Tensor:
    """r_squared delta forward (official loop form): B(M(A(x))) * scaling, (N, out)."""
    return ((x @ a_w.T) @ m.T) @ b_w.T * scaling


def delta_weight(b_w: torch.Tensor, m: torch.Tensor, a_w: torch.Tensor,
                 scaling: float) -> torch.Tensor:
    """Merged delta: B @ M @ A * scaling (fan_in_fan_out=False)."""
    return b_w @ m @ a_w * scaling


def latent_init_grad(s: torch.Tensor) -> torch.Tensor:
    """r_squared grad-mode latent mapping weight: M = diag(Sr)."""
    return torch.diag(s).clone()


def a_near_zero(r: int, in_features: int, seed: int = 0) -> torch.Tensor:
    """Non-r_squared fallback: A initialized with normal_(mean=0, std=1e-5)."""
    torch.manual_seed(seed)
    a = torch.empty(r, in_features, dtype=torch.float64)
    a.normal_(mean=0.0, std=1e-5)
    return a


def solve(mode=None, **kw):
    if mode == "sign_update":
        return sign_update_matrix(kw["grad"], kw["lr"])
    if mode == "svd_grad":
        return svd_grad(kw["x"], kw["rank"], niter=kw.get("niter", 10), seed=kw.get("seed", 42))
    if mode == "svd_grad_nodummy":
        return svd_grad_nodummy(kw["x"], kw["rank"], niter=kw.get("niter", 10), seed=kw.get("seed", 42))
    if mode == "init_ab":
        return init_ab(kw["dec"], kw["enc"], half_init_dec=kw.get("half_init_dec", False),
                       random_init=kw.get("random_init", False), seed=kw.get("seed", 0))
    if mode == "latent_forward":
        return latent_forward(kw["x"], kw["a_w"], kw["m"], kw["b_w"], kw["scaling"])
    if mode == "delta_weight":
        return delta_weight(kw["b_w"], kw["m"], kw["a_w"], kw["scaling"])
    if mode == "latent_init":
        return latent_init_grad(kw["s"])
    if mode == "a_near_zero":
        return a_near_zero(kw["r"], kw["in_features"], seed=kw.get("seed", 0))
    raise ValueError(f"unknown mode {mode!r}")
