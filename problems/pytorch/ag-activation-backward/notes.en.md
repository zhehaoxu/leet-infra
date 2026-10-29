# Notes: Kink Conventions and the Consequences of the Forward Approximation

## Origin

"Give me the derivative of GELU" is a routine rapid-fire question in large-model framework interviews because FFNs are built on it, and "what does relu take at 0" tests whether one understands the **subgradient** rather than "the derivative". Bundling eight activations into one problem mirrors the actual interview rhythm: relu, then leaky, then elu, then gelu/silu.

## Why a Kink Value Is Both Free and Fixed

relu is not differentiable at \(x=0\); autograd uses the right-derivative convention (`torch.relu`'s backward is `where(x > 0, 1, 0)`), so the value there is 0. This does not affect convergence (the point has probability zero) but it **does affect the cross-check**: an implementation using `x >= 0` differs entry by entry on inputs containing 0. elu, in contrast, is genuinely differentiable at 0 (left limit \(e^0=1\), right limit 1), and silu/softplus are analytic, so 0.5 there is a real derivative rather than a convention.

## The Two Forms of gelu Are Not Interchangeable

The exact form is \(x\,\Phi(x)\) with \(\Phi(x)=\tfrac12(1+\operatorname{erf}(x/\sqrt2))\), whose derivative is \(\Phi(x)+x\phi(x)\) for the standard normal density \(\phi\). The tanh approximation \(\tfrac12x(1+\tanh(0.7978(x+0.044715x^3)))\) has a completely different derivative. PyTorch's `F.gelu` defaults to the exact form (`approximate='none'`), so judging uses erf; a hand-written tanh approximation drifts by about \(10^{-3}\) near \(|x|\approx2\) and fails.

## Core Ideas and Pitfalls

- **Do not reuse `sigmoid*(1-sigmoid)` for softplus**: the derivative of softplus is just \(\sigma(x)\); the extra factor breaks it immediately.
- **The leaky_relu kink**: the backward of `F.leaky_relu` is measured to be `x > 0 ? 1 : slope`, so \(x=0\) returns `negative_slope` (0.01 by default), consistent with relu giving 0 at 0.
- **Shape and dtype**: a 0-dim `x` still returns the same dtype; comparisons produce bools that must be cast to float before multiplying `dY`.
- **Validate `kind` before computing**: an unknown kind must raise ValueError instead of silently falling into the relu branch.

## How It Differs from Neighboring Problems

`ag-layernorm-backward` and `ag-rmsnorm-backward` involve reduction-coupled gradients; this problem is strictly elementwise. `ag-lstm-bptt-backward` consumes the same sigmoid/tanh derivatives, where one missing factor corrupts the whole time axis.

## Cross-Check Basis and Complexity

The oracle builds a graph per `F.{relu,leaky_relu,elu,sigmoid,tanh,gelu,silu,softplus}` and takes gradients, atol=1e-9 in float64. The example uses relu over five probe points including 0. Complexity is \(O(\text{numel})\) with no shared intermediates across branches.
