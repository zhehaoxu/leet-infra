# Uncommon Activations: softsign/hardtanh/celu/log-sigmoid

## Problem

Beyond the ReLU/ELU/GELU headline family, torch's activation list holds a set of piecewise/bounded staples: softsign is the rational stand-in for tanh, hardtanh is clamp with hard saturation, celu is the ELU variant with alpha in the exponential's denominator, and log-sigmoid is the numerical foundation of the BCE loss. This problem implements an elementwise activation dispatcher with four modes:

\[\mathrm{softsign}(x)=\frac{x}{1+|x|},\qquad
\mathrm{hardtanh}(x)=\mathrm{clamp}(x,\ \text{min},\ \text{max}),\]
\[\mathrm{celu}(x;\alpha)=\max(0,x)+\min\big(0,\ \alpha\,(e^{x/\alpha}-1)\big),\qquad
\mathrm{logsigmoid}(x)=-\,\mathrm{softplus}(-x).\]

The convention follows the torch official documentation (F.softsign / F.hardtanh / F.celu / F.logsigmoid). The numerical implementation is the heart of the problem: at large alpha the naive \(e^{x/\alpha}-1\) cancels catastrophically and must be replaced by expm1 with a high-precision intermediate; logsigmoid must use the \(-\mathrm{softplus}(-x)\) stable form — the naive \(\log(\mathrm{sigmoid}(x))\) underflows to \(-\infty\) at large negative x. Elementwise, no RNG.

## Rules

- The full signature, parameter meanings and defaults live only in the starter docstring; it is the single judging standard. mode is one of `"softsign"` / `"hardtanh"` / `"celu"` / `"logsigmoid"`, anything else raises; the input is a float tensor (float32/float64 both supported) of arbitrary shape, applied elementwise, returning a tensor of the same shape and dtype.
- **softsign**: \(x/(1+|x|)\); min_value/max_value/alpha are all ignored. Every finite input lands strictly inside \((-1,1)\), with elementwise sign preserved.
- **hardtanh**: clamped into [min_value, max_value] (defaults −1.0 / 1.0); alpha ignored. On the saturated segments the output is **exactly** the boundary value (the judge requires exact equality, not closeness).
- **celu**: \(\max(0,x)+\min(0,\alpha(e^{x/\alpha}-1))\), with the convention \(\alpha>0\); min_value/max_value ignored. The numerical implementation must be mathematically identical to the formula and **always finite**: expm1 plus a high-precision (float64) intermediate, avoiding the catastrophic cancellation of \(e^{x/\alpha}-1\) at large alpha.
- **logsigmoid**: \(-\mathrm{softplus}(-x)\). The stable form is **mandatory**; a direct \(\log(\mathrm{sigmoid}(x))\) is forbidden: for \(x\le-104\) the fp32 sigmoid underflows to 0 and \(\log(0)=-\infty\), while the softplus form is always finite (very negative x drives softplus(−x) onto its linear segment \(\approx-x\), so the result \(\approx x\)).

## Example

Input \(x=[-2,\,-0.5,\,0,\,0.5,\,2]\), default parameters for all four modes (values from the EXAMPLE_* backfills in test_cases.py):

```
softsign   [-0.6666666865, -0.3333333433,  0.0, 0.3333333433,  0.6666666865]
hardtanh   [-1.0,          -0.5,           0.0, 0.5,           1.0         ]
celu(α=1)  [-0.8646647334, -0.3934693336,  0.0, 0.5,           2.0         ]
logsigmoid [-2.1269280910, -0.9740769863, -0.6931471825, -0.4740769863, -0.1269280165]
```

celu's \(-0.8646647=-\,(1-e^{-2})\) (at \(\alpha=1\) the negative branch \(\alpha(e^{x/\alpha}-1)\) is \(e^x-1\)); logsigmoid(0) \(=-\ln2=-0.6931471825\).

## Constraints

- An illegal mode raises ValueError; output shape/dtype match the input; in float64 all four modes match the torch references at 1e-9, float32 at 1e-6
- Discriminating probes asserted directly: softsign is unchanged when handed non-default min/max/alpha, stays strictly inside \((-1,1)\) with sign preserved (at \(\pm10^4\), |y|>0.999 yet still <1); hardtanh's saturated segments equal the boundaries exactly via torch.equal (\([-7,-1,0,1,7]\to[-1,-1,0,1,1]\)), custom boundaries apply and alpha is ignored; celu matches F.celu for \(\alpha\in\{0.5,1.0,2.5\}\) and **degenerates to the identity at \(\alpha=10^6\)** (a naive fp32 implementation zeroes the whole negative branch at small |x| and drifts at the 1e-2 level elsewhere); logsigmoid is finite and matches F.logsigmoid on \(\{-120,-104,-50,0,50,104,120\}\) (the naive form is measured non-finite at −120/−104)
