# Leaky ReLU in Depth

## Definition

$$ \operatorname{leaky}(x)=\begin{cases} x, & x>0\\[3pt] a\,x, & x\le 0\end{cases}\qquad(a=\text{negative\_slope},\ \text{default }0.01) $$

A very small slope is added to the negative half of ReLU, and the piecewise line is continuous at the origin.

## Design motivation

ReLU's gradient is identically 0 in the negative region → dead neurons. Leaky ReLU retains a little gradient flow in the negative region (slope $a$), so a neuron that mistakenly falls into that region still has a chance to "come back to life". $a$ is usually $0.01$; empirical values such as $1/3$ and $0.1$ also exist.

## Implementation & Numerical Notes

- The reference solution is `torch.where(x >= 0, x, negative_slope * x)`, vectorized.
- **Convention**: the split uses $x\ge0$ (the origin follows the positive branch, i.e. the output at the origin is 0); this matches the result of a `torch.maximum`-based ReLU at the origin, so grading is unambiguous.
- Equivalent forms: `x * where(x>0, 1, a)`, or `F.leaky_relu(x, negative_slope)`.
- $a$ is an explicit argument—be sure to use it rather than hard-coding 0.01, otherwise custom slopes outside the default tests will be wrong.

## Place in the family

- PReLU: makes $a$ a **learnable** parameter (possibly different per channel).
- ELU: changes the negative region to saturate exponentially at $-a$, smoother than linear and with a mean closer to 0.
- RReLU: samples $a$ randomly during training as a regularizer.

## Related

In this problem negative_slope is a scalar parameter; SELU is the self-normalizing, fixed-scale variant of ELU—do not confuse the two slope constants.
