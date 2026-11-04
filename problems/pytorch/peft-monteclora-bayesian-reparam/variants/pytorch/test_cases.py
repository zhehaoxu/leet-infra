"""MonteCLoRA test cases — judge-style: check(solve) + string CHECKS."""
import math

import torch

CHECKS = ["check_std_wishart", "check_multivariate", "check_dirichlet",
          "check_combine", "check_kl_terms"]


def check_std_wishart(solve):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    m, r = 3, 4
    std_prior = torch.rand(m, r) * 0.3 - 0.15  # log-scale prior, small
    z_wishart = torch.eye(r).unsqueeze(0).repeat(m, 1, 1)
    for i in range(m):
        z_wishart[i] += 0.1 * torch.randn(r, r)
    z_wishart = (z_wishart + z_wishart.transpose(1, 2)) / 2 + 2 * torch.eye(r)

    sigma = solve(mode="std_to_sigma", std_prior=std_prior)
    assert sigma.shape == (m, r, r), sigma.shape
    want = torch.diag_embed(torch.exp(std_prior))
    assert torch.allclose(sigma, want, atol=1e-12, rtol=0)

    gv = solve(mode="wishart_reparam", std=sigma, z_wishart=z_wishart)
    assert gv.shape == (m, r, r), gv.shape
    mid = sigma @ z_wishart @ sigma.transpose(1, 2)
    want_gv = torch.diag_embed(torch.diagonal(mid, dim1=1, dim2=2))
    assert torch.allclose(gv, want_gv, atol=1e-12, rtol=0)

    # all draws scaled by identity wishart: gv = diag(sigma @ I @ sigma) = sigma^2 diag
    gv_id = solve(mode="wishart_reparam", std=sigma, z_wishart=torch.eye(r).repeat(m, 1, 1))
    want_id = torch.diag_embed(torch.exp(std_prior) ** 2)
    assert torch.allclose(gv_id, want_id, atol=1e-12, rtol=0)

    # positive definite: diagonal entries strictly positive (clip guards)
    assert bool((torch.diagonal(gv, dim1=1, dim2=2) > 0).all())


def check_multivariate(solve):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1)
    m, n, in_f, r = 3, 5, 6, 4
    # spd covariances from the wishart path
    std_prior = torch.rand(m, r) * 0.2
    sigma = solve(mode="std_to_sigma", std_prior=std_prior)
    z_wishart = (torch.eye(r) + 0.05 * torch.randn(m, r, r))
    z_wishart = (z_wishart + z_wishart.transpose(1, 2)) / 2 + 2 * torch.eye(r)
    gv = solve(mode="wishart_reparam", std=sigma, z_wishart=z_wishart)
    gv = gv + torch.eye(r).unsqueeze(0) * 1e-3  # ensure PD for cholesky

    z_mvn = torch.randn(m, n, in_f, r)
    var = solve(mode="multivariate_reparam", z_mvn=z_mvn, cov=gv, sample_scaler=3e-4)
    assert var.shape == (m, in_f, r), var.shape

    L = torch.linalg.cholesky(gv)
    varsum = torch.mean(torch.einsum("mnir,mrk->mnik", z_mvn, L), dim=1)
    assert torch.allclose(var, 3e-4 * varsum, atol=1e-12, rtol=0)

    # scaler respected: double scaler doubles output
    var2 = solve(mode="multivariate_reparam", z_mvn=z_mvn, cov=gv, sample_scaler=6e-4)
    assert torch.allclose(var2, 2 * var, atol=1e-12, rtol=0)

    # non-diagonal SPD covariance (interface-legal; breaks L == L.T shortcuts)
    a_rand = torch.randn(m, r, r)
    cov_nd = a_rand @ a_rand.transpose(1, 2) + torch.eye(r).unsqueeze(0)
    var_nd = solve(mode="multivariate_reparam", z_mvn=z_mvn, cov=cov_nd, sample_scaler=3e-4)
    L_nd = torch.linalg.cholesky(cov_nd)
    want_nd = 3e-4 * torch.mean(torch.einsum("mnir,mrk->mnik", z_mvn, L_nd), dim=1)
    assert torch.allclose(var_nd, want_nd, atol=1e-12, rtol=0)

    # extreme negative-definite z_wishart: clip guards the diagonal at EPS
    z_neg = -1000.0 * torch.eye(r).unsqueeze(0).repeat(m, 1, 1)
    gv_neg = solve(mode="wishart_reparam", std=sigma, z_wishart=z_neg)
    mid_neg = sigma @ z_neg @ sigma.transpose(1, 2)
    clipped = torch.diagonal(mid_neg, dim1=1, dim2=2).clamp(min=1e-6)
    assert torch.allclose(torch.diagonal(gv_neg, dim1=1, dim2=2), clipped, atol=1e-15, rtol=0)
    assert abs(gv_neg[0, 0, 0].item() - 1e-6) < 1e-15, "clip floor must be EPS=1e-6"


def check_dirichlet(solve):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(2)
    m = 4
    prior = torch.rand(m) * 0.5
    z = torch.randn(m)

    alpha = torch.exp(prior)
    out = solve(mode="dirichlet_reparam", alpha=alpha, z_dirichlet=z)
    assert out.shape == (m,), out.shape
    mu = torch.log(alpha) - torch.log(alpha).mean()
    sigma = torch.diag(1.0 / alpha * (1 - 2.0 / m)
                       + 1.0 / (m ** 2) * (1.0 / alpha).sum())
    L = torch.linalg.cholesky(sigma)
    assert torch.allclose(out, L @ z + mu, atol=1e-12, rtol=0)

    # weights: sigmoid then normalized to sum 1
    w = solve(mode="expert_weights", prior=prior, z_dirichlet=z)
    assert w.shape == (m,)
    assert abs(w.sum().item() - 1.0) < 1e-12, w.sum()
    assert bool((w > 0).all()) and bool((w < 1).all())
    raw = torch.sigmoid(out)
    assert torch.allclose(w, raw / raw.sum(), atol=1e-12, rtol=0)

    # zero z reduces to normalized sigmoid of mu
    w0 = solve(mode="expert_weights", prior=prior, z_dirichlet=torch.zeros(m))
    raw0 = torch.sigmoid(mu)
    assert torch.allclose(w0, raw0 / raw0.sum(), atol=1e-12, rtol=0)


def check_combine(solve):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(3)
    m, in_f, r = 4, 6, 3
    weight = torch.randn(in_f, r)          # lora_A.weight.T
    var = torch.randn(m, in_f, r) * 1e-3
    wts = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float64)

    combined = solve(mode="combine_factor", weight=weight, var=var, expert_weights=wts)
    assert combined.shape == (in_f, r), combined.shape
    updated = weight.unsqueeze(0) + 3e-4 * var
    want = torch.einsum("i,ijk->jk", wts, updated)
    assert torch.allclose(combined, want, atol=1e-12, rtol=0)

    # uniform weights reduce to plain mean of perturbed factors
    wu = torch.full((m,), 1.0 / m, dtype=torch.float64)
    combined_u = solve(mode="combine_factor", weight=weight, var=var, expert_weights=wu)
    assert torch.allclose(combined_u, updated.mean(dim=0), atol=1e-12, rtol=0)

    # degenerate weights on one expert pick that expert's factor
    w1 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    combined_1 = solve(mode="combine_factor", weight=weight, var=var, expert_weights=w1)
    assert torch.allclose(combined_1, updated[0], atol=1e-12, rtol=0)

    # NaN in one non-selected expert is nan_to_num-ed away (official behavior)
    var_nan = var.clone()
    var_nan[3, 0, 0] = float("nan")
    combined_n = solve(mode="combine_factor", weight=weight, var=var_nan, expert_weights=w1)
    assert not torch.isnan(combined_n).any(), "nan_to_num must sanitize"


def check_kl_terms(solve):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(4)
    m, n, r = 3, 2, 4
    prior = torch.rand(m) * 0.5
    std_prior = torch.rand(m, r) * 0.2
    gv = torch.eye(r).unsqueeze(0).repeat(m, 1, 1) * 1.5
    wts = torch.full((m,), 1.0 / m, dtype=torch.float64)

    terms = solve(mode="kl_terms", prior=prior, std_prior=std_prior,
                  gaussian_var=gv, expert_weights=wts, m=m, n=n, r=r,
                  dirichlet_prior=1.0, kl_loss_weight=1e-5)
    assert set(terms) == {"dirichlet", "wishart", "multivariate", "entropy", "total"}

    # multivariate_kl on var = c*I: n*0.5*(trace - logdet - d) = n*0.5*(c*r - log(c)*r - r)
    c = 1.5
    want_mv = n * 0.5 * (c * r - math.log(c) * r - r)
    assert abs(terms["multivariate"].item() - want_mv) < 1e-9, (terms["multivariate"], want_mv)

    # wishart_kl on std = diag(exp(std_prior)): var_ii = exp(2*s_i)
    s = torch.exp(std_prior)
    want_w = 0.5 * (-(torch.log(s ** 2).sum(dim=1)) * r + (s ** 2).sum(dim=1) * r - r ** 2)
    assert torch.allclose(terms["wishart"], want_w.mean(), atol=1e-12, rtol=0)

    # entropy is the squared sum (official quirk), NOT Shannon entropy
    assert abs(terms["entropy"].item() - (wts ** 2).sum().item()) < 1e-12
    shannon = -(wts * torch.log(wts)).sum()
    assert abs(terms["entropy"].item() - shannon.item()) > 1e-6

    # uniform m weights: squared sum = 1/m
    assert abs(terms["entropy"].item() - 1.0 / m) < 1e-12

    # total includes kl_loss_weight and the m*n normalization
    want_total = 1e-5 * (terms["dirichlet"] + terms["wishart"]
                         + terms["multivariate"]) / (m * n)
    assert torch.allclose(terms["total"], want_total, atol=1e-15, rtol=0)

    # dirichlet KL with uniform alpha == symmetric prior is zero
    prior_unif = torch.log(torch.ones(m))
    terms0 = solve(mode="kl_terms", prior=prior_unif, std_prior=std_prior,
                   gaussian_var=gv, expert_weights=wts, m=m, n=n, r=r)
    assert abs(terms0["dirichlet"].item()) < 1e-12, terms0["dirichlet"]

    # non-uniform alpha: manual replay of the official dirichlet_kl formula
    alpha2 = torch.exp(prior)
    alpha1 = torch.ones(m, dtype=torch.float64)
    gamma = lambda v: torch.lgamma(v).exp()
    want_dir = (torch.log(gamma(alpha2.sum()) / gamma(alpha1.sum()))
                + (torch.log(gamma(alpha2) / gamma(alpha1))).sum()
                + ((alpha2 - alpha1) * (torch.digamma(alpha2)
                                        - torch.digamma(alpha2.sum()))).sum())
    assert torch.allclose(terms["dirichlet"], want_dir, atol=1e-12, rtol=0), (terms["dirichlet"], want_dir)


def check(solve):
    for name in CHECKS:
        globals()[name](solve)
