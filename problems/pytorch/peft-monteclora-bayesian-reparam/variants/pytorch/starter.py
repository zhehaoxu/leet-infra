"""MonteCLoRA starter — implement solve(mode, **kw) to pass all checks.

Reference: LCS2-IIITD/MonteCLoRA ``peft/src/peft/tuners/tuners_utils_m.py``
(arXiv:2411.04358), class MonteCLoRASampler. All draws (z_mvn, z_wishart,
z_dirichlet) arrive caller-provided — the domain is deterministic.

Modes to implement:
  std_to_sigma           -> diag_embed(exp(std_prior)): (m, r) -> (m, r, r)
  wishart_reparam        -> diag(clip(std @ Z_w @ std.T, min=1e-6))
  multivariate_reparam   -> scaler * mean_n(einsum('mnir,mrk->mnik', z_mvn, L)),
                            L = cholesky(cov)
  dirichlet_reparam      -> L_d @ z + mu_d  (log-normal approximation)
  expert_weights         -> sigmoid(dirichlet) normalized to sum 1
  combine_factor         -> einsum('i,ijk->jk', wts, weight + scaler*var)
  kl_terms               -> {"dirichlet", "wishart", "multivariate",
                             "entropy", "total"}; entropy = squared sum
"""
