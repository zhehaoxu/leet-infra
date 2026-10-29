# Notes: The Two Coupled Paths of the LayerNorm Backward

## Origin

The LayerNorm backward is the standard written test for "implement a fused LN kernel" roles (fused backward kernels come up constantly in Triton/CUDA interviews), and it is where inference frameworks most often break on the reduction order of `dgamma`. What the interviewer really wants to hear is why dX carries two subtractions and which forward step each one comes from.

## Locating the Two Subtractions via the Chain Rule

For one row, \(y_i=\gamma_i x̂_i+\beta_i\) with \(x̂_i=(x_i-\mu)/\sigma\). Writing \(g_i=dY_i\gamma_i\),

$$\frac{\partial L}{\partial x_j}=\sum_i g_i\frac{\partial x̂_i}{\partial x_j},\qquad
\frac{\partial x̂_i}{\partial x_j}=\frac{1}{\sigma}\Big(\delta_{ij}-\frac1H-\frac{(x_i-\mu)(x_j-\mu)}{H\sigma^2}\Big)$$

The \(\delta_{ij}\) term gives \(g/\sigma\); the second term stems from \(\partial\mu/\partial x_j=1/H\) and gives \(-\text{mean}(g)/\sigma\); the third comes from \(\partial\sigma/\partial x_j=(x_j-\mu)/(H\sigma)\) and gives \(-\hat x_j\text{mean}(g\circ\hat x)/\sigma\). **All three survive, the last two with a minus sign.** Dropping the minus on the third term is the classic mistake, and it stays hidden exactly when the forward row is symmetric.

## Core Ideas and Pitfalls

- **Biased variance**: `unbiased=False`, denominator \(H\) not \(H-1\); copying BatchNorm's estimator rescales everything.
- **dGamma/dBeta fold over leading axes**: `(B,T,H)` must sum to `(H,)` using `dim=tuple(range(x.dim()-1))`, not a hard-coded `dim=0`.
- **1-D input**: with `x` of shape `(H,)` the folding set is empty, and the result must stay `(H,)` rather than a scalar.
- **The beta gradient is just the folded dY**, independent of `x` and `gamma` — the test pins this down with "changing beta changes nothing".

## How It Differs from Neighboring Problems

`ag-rmsnorm-backward` has no mean subtraction, so only one subtraction remains in dX; `ag-batchnorm-backward` reduces along the batch axis instead of the feature axis and additionally has to differentiate the batch statistics themselves (a three-way coupling). `ag-weightnorm-backward` is a reparameterization and has nothing to do with normalization axes.

## Cross-Check Basis and Complexity

The oracle calls `F.layer_norm(x, (H,), gamma, beta, eps)` and takes the three gradients with `torch.autograd.grad`, atol=1e-9 in float64. The example uses `eps=0.0` with \(\pm1\) inputs so \(\mu=0,\sigma=1\) and every entry is hand-computable. Complexity is \(O(BTH)\) time: two reductions along the last axis plus one fold over leading axes.
