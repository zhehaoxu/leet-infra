"""MonteCLoRA (arXiv:2411.04358) — Bayesian reparameterization of LoRA factors.

Official account: LCS2-IIITD/MonteCLoRA ``peft/src/peft/tuners/tuners_utils_m.py``
(commit 0cd4a77, class MonteCLoRASampler). All math runs in float64; the
stochastic draws (z_mvn / z_wishart / z_dirichlet) are caller-provided, so the
whole judging domain is deterministic.

Reparameterization chain (official methods):
  std            = diag_embed(exp(std_prior))            # (m, r, r)
  gaussian_var   = diag(clip(std @ Z_wishart @ std.T))   # diag of each matrix
  var            = sample_scaler * mean_n( einsum('mnir,mrk->mnik', z_mvn, L) )
                   with L = cholesky(gaussian_var)       # (m, in, r)
  dirichlet      = L_d @ z_dirichlet + mu_d
                   mu_d  = log(alpha) - mean(log(alpha))
                   sigma = diag( (1-2/M)/alpha + (1/M^2) * sum(1/alpha) )
  expert_weights = sigmoid(dirichlet); expert_weights /= expert_weights.sum()

Combination into the LoRA factor (official layer.py forward):
  updated_A = nan_to_num(A.weight.T + sample_scaler * var)   # per-expert
  updated_A = einsum('i,ijk->jk', expert_weights, updated_A) # weighted mean
  lora_A.weight = updated_A.T                                # standard LoRA forward

Variational loss (official get_variational_loss, weight kl_loss_weight,
normalized by m*n; entropy term is the squared sum, NOT Shannon entropy):
  kl1 = dirichlet_kl(alpha)          vs symmetric prior dirichlet_prior
  kl2 = wishart_kl(diag(exp(std_prior)))
  kl3 = multivariate_kl(gaussian_var)
"""
import torch
import torch.nn.functional as F

EPS = 1e-6


def std_to_sigma(std_prior):
    """diag_embed(exp(std_prior)): (m, r) -> (m, r, r)."""
    return torch.diag_embed(torch.exp(std_prior))


def wishart_reparameterization(std, z_wishart):
    """Official: sigma = diag(clip(std @ Z_wishart @ std^T)). std: (m, r, r),
    z_wishart: (m, r, r) — caller-provided Wishart draw."""
    updated_var = torch.clip(torch.bmm(torch.bmm(std, z_wishart), std.transpose(1, 2)),
                             min=EPS)
    return torch.diag_embed(torch.diagonal(updated_var, dim1=1, dim2=2))


def multivariate_reparameterization(z_mvn, cov_matrix, sample_scaler=3e-4):
    """Official: L = cholesky(cov); varsum = mean_n(einsum('mnir,mrk->mnik', z_mvn, L));
    return sample_scaler * varsum. z_mvn: (m, n, in, r); cov: (m, r, r)."""
    L = torch.linalg.cholesky(cov_matrix)
    varsum = torch.mean(torch.einsum("mnir,mrk->mnik", z_mvn, L), dim=1)
    varsum = torch.nan_to_num(varsum, nan=EPS)
    return sample_scaler * varsum


def dirichlet_reparameterization(alpha, z_dirichlet):
    """Official log-normal approximation. alpha: (m,); z_dirichlet: (m,)."""
    m = alpha.shape[0]
    mu = torch.log(alpha) - torch.log(alpha).mean()
    sigma = torch.diag(1.0 / alpha * (1 - 2.0 / m)
                       + 1.0 / (m ** 2) * (1.0 / alpha).sum())
    L = torch.linalg.cholesky(sigma)
    return L @ z_dirichlet + mu


def expert_weights_from(prior, z_dirichlet):
    """Official: alpha = exp(prior); dirichlet reparam -> sigmoid -> normalize."""
    alpha = torch.exp(prior)
    w = torch.sigmoid(dirichlet_reparameterization(alpha, z_dirichlet))
    return w / w.sum()


def combine_factor(weight, var, expert_weights, sample_scaler=3e-4):
    """Official layer.py combination: per-expert mu + scaler*var, weighted mean.
    weight: (in, r) (i.e. lora_A.weight.T); var: (m, in, r); wts: (m,)."""
    updated = torch.nan_to_num(weight.unsqueeze(0) + sample_scaler * var, nan=0.0)
    return torch.einsum("i,ijk->jk", expert_weights, updated)


def dirichlet_kl(alpha2, dirichlet_prior=1.0):
    """Official dirichlet_kl vs the symmetric prior."""
    m = alpha2.shape[0]
    alpha1 = torch.full((m,), dirichlet_prior, dtype=alpha2.dtype)
    gamma = lambda v: torch.lgamma(v).exp()
    return (torch.log(gamma(alpha2.sum()) / gamma(alpha1.sum()))
            + (torch.log(gamma(alpha2) / gamma(alpha1))).sum()
            + ((alpha2 - alpha1) * (torch.digamma(alpha2)
                                    - torch.digamma(alpha2.sum()))).sum())


def wishart_kl(std):
    """Official wishart_kl on std: (m, r, r). Only diagonal entries are used."""
    var = torch.bmm(std, std.transpose(1, 2))
    diag_values = torch.diagonal(var, dim1=1, dim2=2)
    log_var_trace = torch.sum(torch.log(diag_values), dim=1)
    var_trace = torch.sum(diag_values, dim=1)
    d = std.shape[-1]
    kl_div = 0.5 * (-log_var_trace * d + var_trace * d - d ** 2)
    return kl_div.mean()


def multivariate_kl(var, monteclora_n):
    """Official multivariate_kl: n * 0.5 * (trace - logdet - d), mean over m."""
    var = torch.clamp(var, min=1e-6)
    var_trace = torch.diagonal(var, dim1=1, dim2=2).sum(dim=1)
    log_var_trace = torch.logdet(var)
    d = var.shape[-1]
    kl_div = monteclora_n * 0.5 * (var_trace - log_var_trace - d)
    return kl_div.mean()


def calculate_entropy(expert_weights):
    """Official 'entropy': squared sum of the weights (NOT Shannon entropy)."""
    return (expert_weights ** 2).sum()


def solve(mode, **kw):
    if mode == "std_to_sigma":
        return std_to_sigma(kw["std_prior"])
    if mode == "wishart_reparam":
        return wishart_reparameterization(kw["std"], kw["z_wishart"])
    if mode == "multivariate_reparam":
        return multivariate_reparameterization(kw["z_mvn"], kw["cov"],
                                               kw.get("sample_scaler", 3e-4))
    if mode == "dirichlet_reparam":
        return dirichlet_reparameterization(kw["alpha"], kw["z_dirichlet"])
    if mode == "expert_weights":
        return expert_weights_from(kw["prior"], kw["z_dirichlet"])
    if mode == "combine_factor":
        return combine_factor(kw["weight"], kw["var"], kw["expert_weights"],
                              kw.get("sample_scaler", 3e-4))
    if mode == "kl_terms":
        out = {
            "dirichlet": dirichlet_kl(torch.exp(kw["prior"]),
                                      kw.get("dirichlet_prior", 1.0)),
            "wishart": wishart_kl(std_to_sigma(kw["std_prior"])),
            "multivariate": multivariate_kl(kw["gaussian_var"], kw["n"]),
            "entropy": calculate_entropy(kw["expert_weights"]),
        }
        out["total"] = kw.get("kl_loss_weight", 1e-5) * (out["dirichlet"]
                                                         + out["wishart"]
                                                         + out["multivariate"]) / (kw["m"] * kw["n"])
        return out
    if mode == "full_step":
        # one sampler step, all draws caller-provided
        std = std_to_sigma(kw["std_prior"])
        gaussian_var = wishart_reparameterization(std, kw["z_wishart"])
        var = multivariate_reparameterization(kw["z_mvn"], gaussian_var,
                                              kw.get("sample_scaler", 3e-4))
        w = expert_weights_from(kw["prior"], kw["z_dirichlet"])
        return {"var": var, "expert_weights": w, "gaussian_var": gaussian_var}
    raise ValueError(f"unknown mode: {mode}")


CHECKS = ["check_std_wishart", "check_multivariate", "check_dirichlet",
          "check_combine", "check_kl_terms"]
