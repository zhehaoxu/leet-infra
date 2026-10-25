# GELU Explained

## Definition

Gaussian Error Linear Unit: use the standard normal CDF $\Phi$ as the gate:

$$ \text{gelu}(x) = x\cdot\Phi(x),\qquad \Phi(x)=\tfrac12\big(1+\operatorname{erf}(x/\sqrt2)\big) $$

Intuition: each input is retained weighted by "the probability that it falls in the positive half".

## The Two Approximations (easy to mix up)

1. **Exact erf form** (torch `approximate='none'`): `0.5*x*(1+erf(x/sqrt(2)))`.
2. **tanh approximation** (`approximate='tanh'`):
   $$ 0.5x\Big(1+\tanh\big[\tfrac{\sqrt2}{\sqrt\pi}(x+0.044715x^3)\big]\Big) $$

The two are numerically different (at $x=1$ they differ on the order of $10^{-3}$). **You must align with the correct criterion**—this collection uses the exact erf form; do not gloss over it with the tanh approximation.

## Implementation Notes

- Use `torch.erf`; do not omit the $x/\sqrt2$ factor.
- Large negative $x$: $\Phi\to0$, gelu $\to 0$; large positive $x$: $\Phi\to1$, gelu $\to x$. erf saturates at both ends, so there is no overflow.

## Gradient

$$ \text{gelu}'(x)=\Phi(x)+x\cdot\phi(x),\quad \phi(x)=\tfrac{1}{\sqrt{2\pi}}e^{-x^2/2} $$

GELU is the default activation of the Transformer FFN; its only difference from Swish is that the gate function is the Gaussian CDF rather than the sigmoid.
