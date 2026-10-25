# Softplus Explained

## Definition

The smooth version of ReLU, and the two-term special case of log-sum-exp (LSE):

$$ \text{softplus}(x)=\ln(1+e^{x}) $$

## Key properties

- It is a smooth approximation of $\max(0,x)$: $\approx x$ when $x\gg0$, $\approx 0$ when $x\ll0$.
- Its derivative is exactly the sigmoid: $\text{softplus}'(x)=\sigma(x)$ — it is an antiderivative of sigmoid.
- Always positive, so it is commonly used to parametrize unconstrained variables into quantities that must be positive, such as scale/variance: $\text{scale}=\text{softplus}(\rho)$.

## Numerical stability: the real trap in this level

The naive `log(1+exp(x))` overflows `exp` to inf for large $x$, and log(inf)=inf, losing the true value $\approx x$. The stable equivalent form is:

$$ \text{softplus}(x)=\max(x,0)+\ln\!\big(1+e^{-|x|}\big) $$

because $e^{-|x|}\le 1$ never overflows. In PyTorch, `F.softplus` and `torch.logaddexp(0, x)` both build in this protection.

Scaled by `beta`: $\tfrac1\beta\ln(1+e^{\beta x})$; as $\beta\to\infty$ it converges back to ReLU.
