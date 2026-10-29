# Solution Notes · Analytic Backward of DPO

## A three-step chain rule

Let $x=\beta\Delta$ and $L_b=-\log\sigma(x_b)=\mathrm{softplus}(-x_b)$. Since $\frac{d}{dx}\log\sigma(x)=\sigma(-x)$,

$$\frac{\partial L_b}{\partial x_b}=-\sigma(-x_b).$$

Passing through the two linear maps $x=\beta\Delta$ and $\Delta=(s_w-\text{ref}_w)-(s_l-\text{ref}_l)$:

$$\frac{\partial L_b}{\partial\Delta_b}=-\beta\,\sigma(-x_b),\qquad
\frac{\partial L_b}{\partial s_w}=\frac{\partial L_b}{\partial\Delta_b},\qquad
\frac{\partial L_b}{\partial s_l}=-\frac{\partial L_b}{\partial\Delta_b}.$$

Finally $s_w=\sum_t\text{mask}\cdot\text{logp}$ scatters the sequence gradient back to tokens, and the whole thing is divided by $B$ (the forward is a batch mean). The chain has no nonlinear coupling, so the closed form is short — yet dropping the $1/B$, the $\beta$, either sign, or the mask makes the gradient uniformly wrong by a factor or a flip.

## Why the two sides have opposite signs

$\Delta$ has $+1$ on the chosen logprob and $-1$ on the rejected one. Larger $x$ means smaller loss, so raising the chosen logprob lowers the loss (negative gradient) while raising the rejected logprob raises it (positive gradient). For a given $\Delta$, the per-token gradients on the two sides have equal magnitude and opposite sign — exactly DPO's semantics: what separates the two answers is not any absolute quantity but the change in their **difference**.

## Masked positions must be exactly 0

A padding token's logprob does not enter the sequence sum, so it cannot affect the loss and its gradient must be exactly 0. A common approach is "compute the whole row then multiply by the mask", which is mathematically right; the bug appears when masking is forgotten or mis-broadcast and padding positions receive `dL/dΔ`, becoming a spurious update on padding in the optimizer. The tests check both `grad[masked] == 0` and "changing masked logprobs leaves the loss and other gradients unchanged".

## Mean scaling and batch size

The forward is $\frac1B\sum_b$, so every sample's $\partial L/\partial\Delta_b$ carries $1/B$. Duplicating one sample into a batch of two leaves the loss unchanged but halves each token's gradient — exactly what `loss_scaling` verifies. In custom kernels this factor is the easiest to forget, because the forward "divides by B" while the backward must multiply it into every sample.

## Numerical stability

$\sigma(-x)$ approaches 0 for large positive $x$ and 1 for large negative $x$; compute `sigmoid(-x)` directly and avoid $1-\sigma(x)$ (which loses precision by cancellation for large $x$). The forward should use `softplus(-x)` rather than `log(sigmoid(x))`, which underflows to $-\infty$ for very negative $x$. These are recurring pitfalls in DPO implementations, and they are why this problem allows autograd cross-checking but demands the analytic form: the closed form can be embedded in a fused kernel and serves as an independent check against framework gradients.
