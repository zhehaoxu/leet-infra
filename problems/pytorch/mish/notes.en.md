# Mish in Detail

## Definition

$$ \text{mish}(x) = x\cdot\tanh\!\big(\text{softplus}(x)\big),\qquad \text{softplus}(x)=\ln(1+e^{x}) $$

## Design motivation

Like Swish, it is of the form "$x$ × smooth gate", but the gate is $\tanh(\text{softplus}(x))$:

- It is smooth everywhere **and its first derivative is also continuous** (Swish's higher-order derivatives are not), which is said to help optimization.
- It has a shallow valley in the negative range (about $-0.31$ at $x\approx-1.19$), preserving negative signal without saturating.

## Key numerical stability points

- Computing softplus directly as `torch.log1p(torch.exp(x))` overflows in `exp` for large $x$. The stable form is:
  $$ \text{softplus}(x)=\max(x,0)+\ln\!\big(1+e^{-|x|}\big) $$
  torch provides `F.softplus`, or the grading convention uses `torch.logaddexp(0, x)`.
- Large positive $x$: softplus$\to x$, $\tanh(x)\to1$, mish$\to x$; large negative $x$: softplus$\to0$, $\tanh\to0$, mish$\to0$. The behavior at both ends matches intuition.

## Comparison with neighboring activations

| Activation | Gate | Negative range |
|---|---|---|
| ReLU | step | constantly 0 |
| Swish/SiLU | $\sigma(x)$ | shallow valley |
| GELU | $\Phi(x)$ | shallow valley |
| Mish | $\tanh(\text{softplus}(x))$ | slightly deeper valley, smoother second derivative |
