# Notes

- Source: MonteCLoRA (arXiv:2411.04358), official LCS2-IIITD/MonteCLoRA `peft/src/peft/tuners/tuners_utils_m.py`, class `MonteCLoRASampler` (commit 0cd4a77).
- Judging domain: the deterministic core of the reparameterization chain (z_mvn/z_wishart/z_dirichlet all caller-provided) + expert-weight normalization + factor combination + the three KL terms + the squared-sum "entropy"; Wishart Bartlett sampling, the async pre-generation thread, and the MVN distribution objects are out of scope.
- Modes (8): std_to_sigma / wishart_reparam / multivariate_reparam / dirichlet_reparam / expert_weights / combine_factor / kl_terms / full_step.
- Official traps: expert weights are sigmoid-then-normalized (NOT softmax); the "entropy" is the squared sum of weights (NOT Shannon entropy); wishart_kl uses only the diagonal of std@stdᵀ; Dirichlet uses the log-normal approximation (chol(sigma)@z + mu); clip floor 1e-6.
- Anchors in float64; torch.logdet's LU error is ~2.7e-12, so the multivariate_kl analytic comparison uses tolerance 1e-9, everything else 1e-12.
- mut 17: exp omission / diag removal / clip floor / diagonal flip / chol→cov / mean→sum / einsum subscripts / scaler / mu centering / (1-2/M) / sigmoid / normalization / combination scaler / weighted→mean / dirichlet formula / entropy formula / loss normalization — all CAUGHT.
- Lessons: mutation anchors must not hit the docstring (the scaler mutation first replaced the docs); within the official domain cov is always diagonal, making the chol transpose a true equivalence — a non-diagonal SPD cov must be constructed within the interface's legal range to catch the einsum-subscript mutation.
