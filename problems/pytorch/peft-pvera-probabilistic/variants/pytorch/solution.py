"""PVeRA (Probabilistic Vector-based Random Matrix Adaptation, arXiv:2512.07703, PEFT v0.20.0 tuners/pvera).

Shared frozen projections A (2r rows) / B (r cols) across layers; per-layer trainable
lambda_d (2r,) — first half scales the mu path, second half the logvar path — and lambda_b (out,).
Forward: mu, logvar = (lambda_d * x @ A_s.T).chunk(2); z = mu (+ eps*exp(.5*logvar) when sampling);
delta = lambda_b * (z @ B_s.T). No scaling term. Merge uses the mu half only:
dW = (lambda_b * B_s) @ (lambda_d_mu * A_mu). Setup: lambda_d = full(d_initial), lambda_b = ones.
"""
import math

import torch


def shared_projections(r, in_max, out_max, prng_key):
    """Official init: kaiming_uniform_ (default a=0) on float32 empty tensors with CPU generator.
    A has 2r rows (mu/logvar double bank), B has r cols. Returns float64 views plus float32 originals."""
    gen = torch.Generator().manual_seed(prng_key)
    a32 = torch.nn.init.kaiming_uniform_(torch.empty(r * 2, in_max, dtype=torch.float32), generator=gen)
    b32 = torch.nn.init.kaiming_uniform_(torch.empty(out_max, r, dtype=torch.float32), generator=gen)
    return a32.to(torch.float64), b32.to(torch.float64), a32, b32


def slice_pair(a_shared, b_shared, in_features, out_features):
    """Slice per-layer submatrices out of the shared max-size banks."""
    return a_shared[:, :in_features], b_shared[:out_features]


def reparametrize(mu, logvar, eps):
    """eps=None -> eval path z=mu; else z = mu + eps * exp(0.5*logvar)."""
    if eps is None:
        return mu
    return mu + eps * torch.exp(0.5 * logvar)


def forward_delta(x, lambda_d, lambda_b, a_s, b_s, eps=None):
    """x (..., in); lambda_d (2r,); lambda_b (out,); a_s (2r, in); b_s (out, r).
    Adapter increment (base excluded)."""
    pre = lambda_d * (x @ a_s.T)              # (..., 2r)
    mu, logvar = pre.chunk(2, dim=-1)         # (..., r) each
    z = reparametrize(mu, logvar, eps)
    return lambda_b * (z @ b_s.T)             # (..., out)


def merge_delta(lambda_d, lambda_b, a_s, b_s):
    """Mu-half-only merge: dW (out, in) = (lambda_b * B_s) @ (lambda_d_mu * A_mu)."""
    r = lambda_d.shape[0] // 2
    return (lambda_b.unsqueeze(-1) * b_s) @ (lambda_d[:r].unsqueeze(-1) * a_s[:r])


def layer_setup(r, out_features, d_initial=0.1):
    """Official reset account (init_weights=True): lambda_d zeros_->fill(d_initial) (2r,);
    lambda_b zeros_ -> 0 (the torch.ones creation in update_layer is overwritten by reset).
    init_weights=False skips reset: lambda_d keeps torch.randn (unanchorable), lambda_b stays ones."""
    return {
        "lambda_d": torch.full((r * 2,), d_initial, dtype=torch.float64),
        "lambda_b": torch.zeros(out_features, dtype=torch.float64),
        "lambda_b_uninit": torch.ones(out_features, dtype=torch.float64),
    }


def solve(mode=None, **kw):
    if mode == "shared_projections":
        return shared_projections(kw["r"], kw["in_max"], kw["out_max"], kw["prng_key"])
    if mode == "slice_pair":
        return slice_pair(kw["a_shared"], kw["b_shared"], kw["in_features"], kw["out_features"])
    if mode == "forward_delta":
        return forward_delta(kw["x"], kw["lambda_d"], kw["lambda_b"], kw["a_s"], kw["b_s"],
                             eps=kw.get("eps"))
    if mode == "merge_delta":
        return merge_delta(kw["lambda_d"], kw["lambda_b"], kw["a_s"], kw["b_s"])
    if mode == "layer_setup":
        return layer_setup(kw["r"], kw["out_features"], kw.get("d_initial", 0.1))
    raise ValueError(f"unknown mode {mode!r}")
