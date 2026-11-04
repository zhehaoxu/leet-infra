"""Checks for peft-lorasb-sign-svd (judge contract: string CHECKS + check(solve))."""
import math

import torch

torch.set_default_dtype(torch.float64)


def check_sign_update(solve):
    grad = torch.tensor([[2.0, -3.0, 0.0], [0.5, -0.5, 0.0]])
    out = solve(mode="sign_update", grad=grad, lr=0.01)
    expect = -0.01 * torch.sign(grad)
    assert torch.equal(out, expect), out
    # sign(0) = 0 and -lr scaling
    assert out[0, 2] == 0.0 and out[0, 0] == -0.01 and out[0, 1] == 0.01
    # lr = 0 -> all zeros
    assert torch.equal(solve(mode="sign_update", grad=grad, lr=0.0), torch.zeros(2, 3))
    # any nonzero magnitude collapses to +-lr
    grad2 = torch.tensor([[1e-9, -1e9]])
    assert torch.equal(solve(mode="sign_update", grad=grad2, lr=0.1), torch.tensor([[-0.1, 0.1]]))


def check_svd_grad_repro(solve):
    torch.manual_seed(0)
    d_out, d_in, r = 6, 8, 3
    grad = torch.randn(d_out, d_in)
    X = (-0.01 * torch.sign(grad)).T  # official caller passes (in, out)

    enc1, dec1, s1 = solve(mode="svd_grad", x=X, rank=r, niter=10, seed=42)
    enc2, dec2, s2 = solve(mode="svd_grad", x=X, rank=r, niter=10, seed=42)
    assert torch.equal(enc1, enc2) and torch.equal(dec1, dec2) and torch.equal(s1, s2)
    assert enc1.shape == (d_in, r) and dec1.shape == (r, d_out) and s1.shape == (r,)
    # judging contract: outputs are explicitly promoted to float64
    assert enc1.dtype == torch.float64 and dec1.dtype == torch.float64 and s1.dtype == torch.float64

    # singular values sorted descending (svd_lowrank contract)
    assert bool(torch.all(s1[:-1] >= s1[1:]))

    # B@A reconstructs the sign update (out, in); coarse by design but bounded
    b_w, a_w = dec1.T, enc1.T
    recon = b_w @ a_w
    err = (recon - X.T).norm() / X.T.norm()
    assert err < 0.6, err

    # rank-1 input reconstructs exactly
    X1 = (torch.ger(torch.tensor([1.0, 2.0, -1.0, 0.5]),
                    torch.tensor([2.0, -1.0, 3.0]))).T  # (in=3, out=4)
    enc, dec, s = solve(mode="svd_grad", x=X1, rank=1, niter=10, seed=42)
    recon1 = dec.T @ enc.T
    rel = (recon1 - X1.T).norm() / X1.T.norm()
    assert rel < 1e-3, rel

    # value anchors: the official float32 pipeline (incl. dummy-R RNG draw) is deterministic
    s_anchor = torch.tensor([0.04752158001065254, 0.02828427217900753, 0.028284268453717232])
    assert torch.equal(s1, s_anchor), s1
    assert enc1[0, 0] == 0.018178317695856094 and dec1[0, 0] == -0.16099095344543457

    # niter sensitivity: niter=0 (projection only) differs from niter=10
    enc0, dec0, s0 = solve(mode="svd_grad", x=X, rank=r, niter=0, seed=42)
    assert not torch.equal(s0, s1)

    # rank must be positive
    try:
        solve(mode="svd_grad", x=X, rank=0, niter=10, seed=42)
        raise AssertionError("expected ValueError for rank=0")
    except ValueError:
        pass


def check_dummy_r_quirk(solve):
    torch.manual_seed(1)
    X = torch.randn(8, 6)  # (in, out)
    enc_d, dec_d, s_d = solve(mode="svd_grad", x=X, rank=3, niter=10, seed=42)
    enc_n, dec_n, s_n = solve(mode="svd_grad_nodummy", x=X, rank=3, niter=10, seed=42)
    # the dummy RNG draw MUST shift the internal projection: results differ
    assert not torch.equal(s_d, s_n), "dummy R quirk not observable — official flow changed?"
    # each path is individually reproducible
    enc_d2, dec_d2, s_d2 = solve(mode="svd_grad", x=X, rank=3, niter=10, seed=42)
    assert torch.equal(s_d, s_d2) and torch.equal(enc_d, enc_d2)


def check_init_ab(solve):
    torch.manual_seed(2)
    d_out, d_in, r = 6, 8, 4
    grad = torch.randn(d_out, d_in)
    X = (-0.01 * torch.sign(grad)).T
    enc, dec, s = solve(mode="svd_grad", x=X, rank=r, niter=10, seed=42)

    # plain split: A_w = enc.T (r, in), B_w = dec.T (out, r), no RNG consumed
    a_w, b_w = solve(mode="init_ab", dec=dec, enc=enc, seed=-1)
    assert torch.equal(a_w, enc.T) and torch.equal(b_w, dec.T)
    assert a_w.shape == (r, d_in) and b_w.shape == (d_out, r)

    # half_init_dec: lower ceil(r/2) rows of dec re-initialized with kaiming a=sqrt(5)
    a_h, b_h = solve(mode="init_ab", dec=dec, enc=enc, half_init_dec=True, seed=7)
    lower = math.ceil(r / 2)
    # dec rows are b_h COLUMNS after transposing: b_h[:, :lower] untouched
    assert torch.equal(b_h[:, :lower], dec[:lower, :].T)
    assert not torch.equal(b_h[:, lower:], dec[lower:, :].T)  # lower rows changed
    fan_in = d_out
    bound = math.sqrt(6.0 / ((1.0 + 5.0) * fan_in))
    assert b_h[:, lower:].abs().max() <= bound + 1e-9
    assert b_h[:, lower:].abs().max() > 1e-3
    # reproducible with same seed
    a_h2, b_h2 = solve(mode="init_ab", dec=dec, enc=enc, half_init_dec=True, seed=7)
    assert torch.equal(a_h, a_h2) and torch.equal(b_h, b_h2)

    # random_init overrides everything (enc first, then dec)
    a_r, b_r = solve(mode="init_ab", dec=dec, enc=enc, random_init=True, seed=7)
    assert not torch.equal(a_r, enc.T) and not torch.equal(b_r, dec.T)
    # kaiming fan_in is that of the PRE-transpose matrices: enc (in, r) -> fan_in=r,
    # dec (r, out) -> fan_in=out
    bound_a = math.sqrt(6.0 / ((1.0 + 5.0) * r))
    bound_b = math.sqrt(6.0 / ((1.0 + 5.0) * d_out))
    assert a_r.abs().max() <= bound_a + 1e-9 and b_r.abs().max() <= bound_b + 1e-9

    # half_init + random_init: random_init overwrites every VALUE, but half init
    # already advanced the global RNG stream -> the drawn numbers differ (official order)
    a_hr, b_hr = solve(mode="init_ab", dec=dec, enc=enc, half_init_dec=True, random_init=True, seed=7)
    assert not torch.equal(a_hr, a_r) or not torch.equal(b_hr, b_r), \
        "half_init must leave an RNG-stream footprint"
    assert a_hr.abs().max() <= bound_a + 1e-9 and b_hr.abs().max() <= bound_b + 1e-9
    a_hr2, b_hr2 = solve(mode="init_ab", dec=dec, enc=enc, half_init_dec=True, random_init=True, seed=7)
    assert torch.equal(a_hr, a_hr2) and torch.equal(b_hr, b_hr2)

    # random_init order contract: enc drawn BEFORE dec on the shared stream.
    # Replayed on contiguous inputs (svd_grad outputs are non-contiguous views whose
    # uniform_ fill order follows memory layout, so only the bound check applies there).
    torch.manual_seed(9)
    enc_c = torch.randn(7, 5)
    dec_c = torch.randn(5, 6)
    a_c, b_c = solve(mode="init_ab", dec=dec_c, enc=enc_c, random_init=True, seed=11)
    torch.manual_seed(11)
    e_ref = torch.empty(7, 5).uniform_(-math.sqrt(6.0 / 6.0 / 5), math.sqrt(6.0 / 6.0 / 5))
    d_ref = torch.empty(5, 6).uniform_(-math.sqrt(6.0 / 6.0 / 6), math.sqrt(6.0 / 6.0 / 6))
    assert torch.equal(a_c, e_ref.T) and torch.equal(b_c, d_ref.T)

    # odd rank: ceil(rows/2) vs rows//2 differ — official uses ceil
    dec5 = torch.randn(5, 6)
    enc5 = torch.randn(7, 5)
    a_o, b_o = solve(mode="init_ab", dec=dec5, enc=enc5, half_init_dec=True, seed=3)
    assert torch.equal(b_o[:, :3], dec5[:3, :].T)          # ceil(5/2)=3 untouched columns
    assert not torch.equal(b_o[:, 3:], dec5[3:, :].T)      # rows 3..4 re-initialized


def check_latent(solve):
    torch.manual_seed(3)
    d_out, d_in, r = 5, 7, 3
    a_w = torch.randn(r, d_in)
    b_w = torch.randn(d_out, r)
    m = torch.randn(r, r)  # trained latent mapping is a general (asymmetric) matrix
    x = torch.randn(4, d_in)
    scaling = 2.0

    fwd = solve(mode="latent_forward", x=x, a_w=a_w, m=m, b_w=b_w, scaling=scaling)
    d_w = solve(mode="delta_weight", b_w=b_w, m=m, a_w=a_w, scaling=scaling)
    assert torch.allclose(fwd, x @ d_w.T, atol=1e-10), (fwd - x @ d_w.T).abs().max()
    assert d_w.shape == (d_out, d_in) and fwd.shape == (4, d_out)
    assert torch.equal(d_w, b_w @ m @ a_w * scaling)

    # grad-mode latent init: M = diag(Sr)
    _, _, s = solve(mode="svd_grad", x=torch.randn(7, 5), rank=3, niter=10, seed=42)
    m_init = solve(mode="latent_init", s=s)
    assert torch.equal(m_init, torch.diag(s))

    # full chain: init_ab -> delta_weight uses the SAME B@A basis
    X = torch.randn(5, 7).T  # (in, out)
    enc, dec, s = solve(mode="svd_grad", x=X, rank=3, niter=10, seed=42)
    a_w2, b_w2 = solve(mode="init_ab", dec=dec, enc=enc, seed=-1)
    m2 = solve(mode="latent_init", s=s)
    d_w2 = solve(mode="delta_weight", b_w=b_w2, m=m2, a_w=a_w2, scaling=1.0)
    # with M = diag(s): B diag(s) A = V diag(s) diag(s) U^T (spectral-squared recon)
    expect = dec.T @ torch.diag(s) @ enc.T
    assert torch.allclose(d_w2, expect, atol=1e-10)


def check_a_near_zero(solve):
    a = solve(mode="a_near_zero", r=3, in_features=8, seed=11)
    assert a.shape == (3, 8)
    assert a.abs().max() <= 5e-5, a.abs().max()  # normal(0, 1e-5): |a| < ~5 sigma
    assert a.abs().max() > 0
    a2 = solve(mode="a_near_zero", r=3, in_features=8, seed=11)
    assert torch.equal(a, a2)  # seed replay
    a3 = solve(mode="a_near_zero", r=3, in_features=8, seed=12)
    assert not torch.equal(a, a3)


CHECKS = [
    "check_sign_update",
    "check_svd_grad_repro",
    "check_dummy_r_quirk",
    "check_init_ab",
    "check_latent",
    "check_a_near_zero",
]
