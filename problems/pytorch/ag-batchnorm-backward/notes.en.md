# Notes: The Three Terms and the 1/N Accounting in BatchNorm Backward

## Origin

The BatchNorm backward is a classic hand-derivation question for training-framework roles and is also the reproduction of the three formulas in the appendix of the *Batch Normalization* paper. What separates candidates is never memorizing the formula but explaining where the \(1/N\) comes from and why the output must vanish when `N=1`.

## Where the Three Terms and Their 1/N Come From

For a fixed feature column, let \(a_n=dY_n\gamma_n\), \(\hat x_n=(x_n-\mu)/\sigma\), \(\mu=\frac1N\sum_k x_k\), \(\sigma=(\frac1N\sum_k(x_k-\mu)^2+eps)^{1/2}\). Then

$$\frac{\partial L}{\partial x_j}=\frac{a_j}{\sigma}-\frac{1}{\sigma^3}\Big(\frac1N\sum_k a_k\Big)(x_j-\mu)-\frac{1}{\sigma^3}(x_j-\mu)\frac1N\sum_k a_k\hat x_k$$

The first term passes through; the second comes from \(\partial\mu/\partial x_j=1/N\); the third from \(\partial\sigma/\partial x_j=(x_j-\mu)/(N\sigma)\). **Both non-pass-through terms carry exactly one \(1/N\)**, which writing them as \(\text{mean}_N(\cdot)\) supplies automatically. Mixing sum and mean (dividing by N one time too few) is the mirror image of the double-counting bug: the coefficient is off by a factor of N and is exposed immediately by the `eps=0` self-check \(\sum_N dX=0\).

## Core Ideas and Pitfalls

- **Training vs inference**: inference uses `running_mean/running_var`, which are constant w.r.t. `x`, so \(dX=dY\gamma/\sigma_{run}\) has only one term; this problem is about training mode.
- **N=1**: \(\mu=x\), \(var=0\); with `eps>0` the three terms cancel to zero, and the test uses this to pin down self-consistency.
- **dGamma/dBeta sum over the batch (no mean)**: shape `(C,)`, in the opposite direction from LayerNorm's fold over leading axes.
- **beta never enters dX**: perturbing `beta` must leave all three returns bit-identical.
- **Biased variance**: PyTorch's BN divides by \(1/N\), not \(1/(N-1)\).

## How It Differs from Neighboring Problems

`ag-layernorm-backward` and `ag-rmsnorm-backward` normalize along the last feature axis independently per sample, so no cross-sample coupling appears; here normalization is along the batch axis, so one sample perturbs the whole column — the very reason for three terms. Forward-only BatchNorm exercises only compute the output and the running-statistic update.

## Cross-Check Basis and Complexity

The oracle calls `F.batch_norm(x, None, None, gamma, beta, training=True, eps=eps)` and takes the three gradients via `torch.autograd.grad`, atol=1e-9 in float64. The example sets `eps=0` with \(\pm1\) inputs so \(\mu=0,\sigma=1\) and every value is hand-computable. Complexity is \(O(NC)\): two reductions along the batch axis plus one per-column broadcast.
