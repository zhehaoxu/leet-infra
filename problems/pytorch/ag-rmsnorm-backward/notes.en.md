# Notes: Why Only One Subtraction Survives in the RMSNorm Backward

## Origin

RMSNorm is the standard normalization of the LLaMA/Mistral/Gemma family (forward-only variants also exist in this bank). It comes up backward in two interview settings: writing a fused residual+RMSNorm kernel, and explaining which reduction RMSNorm saves and why the gradient form changes. The second question is the best discriminator of whether someone actually derived it.

## Deriving dX from Scale Invariance

With \(y_i=\gamma_i x_i/r\) and \(r=\sqrt{\frac1H\sum_k x_k^2+eps}\), RMSNorm penalizes only perturbations **along \(x\)**: \(\partial r/\partial x_j=x_j/(Hr)\), hence

$$\frac{\partial L}{\partial x_j}=\frac{g_j}{r}-\sum_i g_i\frac{x_i}{r^2}\frac{x_j}{Hr}=\frac1r\Big(g_j-\hat x_j\cdot\text{mean}(g\circ\hat x)\Big)$$

Compared with LayerNorm's \(\frac1\sigma\big(g-\text{mean}(g)-\hat x\,\text{mean}(g\circ\hat x)\big)\), the missing \(\text{mean}(g)\) is exactly the \(\partial\mu/\partial x_j\) path. Geometrically LayerNorm projects the gradient onto the subspace orthogonal to both the all-ones vector and \(x̂\), while RMSNorm removes only the single direction \(x̂\) (i.e. \(x\) itself).

## Core Ideas and Pitfalls

- **Copying the LayerNorm formula**: after the extra \(\text{mean}(g)\), constant \(dY\) along the last axis with all-ones \(\gamma\) wrongly yields 0, whereas the correct \((g-\hat x\,\text{mean}(g\circ\hat x))/r\) is generally nonzero. The test uses exactly this input to separate the two formulas.
- **Definition of \(x̂\)**: here \(x̂=x/rms\) with no centering; feeding a centered residual injects a first-order error.
- **Zero input**: with all-zero \(x\) and \(eps=0\) we get \(r=0\); the default `eps` (1e-6) must guard it so no `nan` appears.
- **dGamma folding**: identical to LayerNorm — sum every leading axis down to `(H,)`.

## How It Differs from Neighboring Problems

`ag-layernorm-backward` has the extra \(\text{mean}(g)\) term and a centered \(x̂\); `ag-batchnorm-backward` reduces along the batch axis where the statistics are themselves differentiable; `ag-weightnorm-backward` is a weight reparameterization with no normalization axis. The signature difficulty here is that the missing term produces an observable nonzero gradient difference.

## Cross-Check Basis and Complexity

The oracle builds the forward in pure autograd as `x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma` and takes `dX/dGamma`, atol=1e-9 in float64. The example uses \(x=\pm1\) and \(eps=0\) so \(rms=1\) and all four entries are hand-computable. Complexity is \(O(BTH)\): one sum-of-squares reduction, one last-axis dot-product reduction and one fold over leading axes.
