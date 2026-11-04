# MonteCLoRA: reducing low-rank estimator variance with Bayesian reparameterization

LoRA is hyperparameter-sensitive and produces high-variance fine-tuned outputs. MonteCLoRA (arXiv:2411.04358, official LCS2-IIITD/MonteCLoRA) parameterizes the low-rank factors as distributions: samples are drawn from the posterior each step, the forward pass is Monte Carlo averaged, and the reparameterization trick keeps gradients unbiased — all with only O(m+r) extra parameters, substantially reducing estimator variance.

## The reparameterization chain (official `MonteCLoRASampler`)

With `std_prior: (m, r)` (log scale) and three external randomness sources `z_wishart: (m,r,r)`, `z_mvn: (m,n,in,r)`, `z_dirichlet: (m,)`:

```python
std = diag_embed(exp(std_prior))            # (m, r, r)
gaussian_var = diag(clip(std @ Z_w @ std.T))  # diagonal Wishart shrinkage, clip floor 1e-6
L = cholesky(gaussian_var)
var = scaler * mean_n(einsum('mnir,mrk->mnik', z_mvn, L))   # (m, in, r)
mu  = log(alpha) - mean(log(alpha))          # alpha = exp(prior)
sigma = diag((1 - 2/M)/alpha + (1/M^2)·sum(1/alpha))
d   = cholesky(sigma) @ z_dirichlet + mu     # log-normal approximation of Dirichlet
w   = sigmoid(d); w = w / w.sum()            # expert weights: sigmoid then normalize, NOT softmax
```

Combination back into the LoRA factor (official `layer.py`):

```python
updated = nan_to_num(A.T + scaler * var)    # per-expert perturbation
A_new   = einsum('i,ijk->jk', w, updated)   # expert-weighted mean
# then the standard LoRA forward: B(A_new(dropout(x))) * scaling
```

## Variational regularization (get_variational_loss)

`kl_loss_weight·(kl_dir + kl_wishart + kl_mv)/(m·n)`, plus an "entropy" term — **the official entropy is the squared sum of weights, NOT Shannon entropy**, an easy trap:

- `multivariate_kl = n·0.5·(trace(var) − logdet(var) − d)`, averaged over m;
- `wishart_kl = 0.5·(−log_trace·d + var_trace·d − d²)`, using only the diagonal of `std@stdᵀ`;
- `dirichlet_kl` is the standard gamma/digamma formula against the symmetric `dirichlet_prior`.

## Judging domain

The sampling distributions themselves (Wishart Bartlett decomposition, the async pre-generation thread) are out of scope — all three z draws are caller-provided, turning the whole reparameterization chain into a deterministic pure function: std→sigma, Wishart shrinkage, Cholesky noise contraction, the Dirichlet approximation, expert-weight normalization, factor combination, and the three KL terms.
