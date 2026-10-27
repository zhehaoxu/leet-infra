# In Depth: Uncommon Activations: softsign/hardtanh/celu/log-sigmoid

## Source

- All four operators follow the torch official documentation: `torch.nn.functional.softsign` / `hardtanh` / `celu` / `logsigmoid` (including hardtanh's min_value/max_value boundary parameters and celu's alpha semantics).
- The original publications of softsign and celu are not on this problem's verified list — **to be re-sourced**; only the torch documentation convention is asserted here, with no unverified identifiers cited.
- The logsigmoid stable form and the softplus threshold semantics (beta=1, linear segment for z>20) follow the torch documentation; F.logsigmoid itself is implemented along the \(-\mathrm{softplus}(-x)\) route.

## Full derivation and index convention

Elementwise operators, no index convention; arbitrary shapes, all four modes preserve shape and dtype. Mode by mode:

**The shape of softsign**: \(x/(1+|x|)\). Sign preservation is obvious (the denominator is positive); the range: for \(x\ge0\), \(y=x/(1+x)=1-\frac{1}{1+x}<1\); for \(x<0\), \(y=x/(1-x)=-1+\frac{1}{1-x}>-1\) — **any finite input lands strictly inside \((-1,1)\)**, approaching the boundary at speed \(1-1/|x|\); contrast hardtanh, which is **exactly** \(\pm1\) once \(|x|\ge1\). It is also the rational stand-in for tanh: no exponential, far slower saturation (tanh(4)=0.9993, softsign(4)=0.8). The probe checks |y|>0.999 yet strictly <1 at \(\pm10^4\).

**The three segments of hardtanh**: clamp is \(\min(\max(x,\text{min}),\text{max})\); slope 1 on the open interval (min,max), 0 outside — cheaper than sigmoid/tanh (no exp), gradients neither vanish nor explode: a "bounded ReLU". The judge requires the saturated segments to be **exactly** the boundary (torch.equal: \([-7,-1,0,1,7]\to[-1,-1,0,1,1]\)): clamp's output is the boundary float itself with no arithmetic, so any approximated saturation (e.g. multiplicative compression) fails.

**celu's segments and the large-alpha limit**: the positive branch \(\max(0,x)\) is the identity; on the negative branch \(\alpha(e^{x/\alpha}-1)<0\), equal to 0 at \(x=0\) with derivative \(e^0=1\) — C¹ continuity at the origin. The difference from ELU is where alpha sits: ELU's negative branch is \(\alpha(e^x-1)\) (alpha outside the exponential, branch shape fixed), celu's is \(\alpha(e^{x/\alpha}-1)\) (alpha in the denominator) — which yields the **identity limit**:

\[\alpha\big(e^{x/\alpha}-1\big)=x+\frac{x^2}{2\alpha}+O(\alpha^{-2})\ \xrightarrow{\alpha\to\infty}\ x,\]

the branch straightens toward a line as alpha grows and the whole function degenerates to the identity. The \(\alpha=10^6\) case walks exactly this line: the true value \(\mathrm{celu}(-5;10^6)=-4.9999875\) (within \(1.25\times10^{-5}\) of −5), and both the judge's atol=1e-4 identity check and the 1e-6 F.celu comparison anchor on it.

**celu's numerical trap (the heart of this problem)**: the naive fp32 \(e^{x/\alpha}-1\) fails in two ways at \(\alpha=10^6\):

1. **The whole negative branch vanishes**: for \(|x/\alpha|\lesssim3\times10^{-8}\) (i.e. \(|x|\lesssim0.03\)), \(e^z\) rounds to **exactly 1.0** in fp32, \(e^z-1\) evaluates to 0, and the branch disappears — measured at \(x=-10^{-3}\): naive output 0.0 versus the true \(-0.001\), an error equal to the entire branch (100%).
2. **Cancellation eats the digits, then ×α amplifies**: fp32's spacing near 1 is \(2^{-24}\approx5.96\times10^{-8}\), while \(e^{-5\times10^{-6}}\) sits only \(5\times10^{-6}\) from 1 — subtracting 1 cancels the significant digits, and the residue times \(\alpha=10^6\) becomes a visible absolute error: measured at \(x=-5\), naive gives \(-5.00679\) (true \(-4.9999875\), error \(6.8\times10^{-3}\)); at \(x=-0.5\) the error is \(2.3\times10^{-2}\).

Two fixes: ① replace \(e^z-1\) with `expm1(z)` (relative accuracy preserved as z→0); ② carry intermediates in float64 and cast back, removing the residual rounding of z itself and of the ×α step. The stable implementation matches F.celu pointwise at atol=1e-6; the naive one cannot even pass the α=1e6 identity check at atol=1e-4.

**The logsigmoid stable form**: \(\log\sigma(x)=x-\log(1+e^x)\) is ill-conditioned in both directions: large positive x overflows \(\log(1+e^x)\); large negative x underflows \(\sigma(x)\). The identity

\[\log\sigma(x)=-\log\big(1+e^{-x}\big)=-\mathrm{softplus}(-x)\]

moves both pathologies onto softplus(−x): as \(x\to-\infty\) the input \(-x\to+\infty\) and softplus rides its linear segment (torch's softplus returns z directly for z>20), so the result \(\approx x\), exactly finite; as \(x\to+\infty\), softplus(−x)→0 and the underflow is harmless. Per the problem's stated convention, fp32 sigmoid underflows to 0 for \(x\le-104\); the probes take −120 and −104, where the naive form is measured to be −inf and the stable form returns exactly −120 and −104 (linear segment). **The linear segment is not optional**: softplus(50) \(=50+2\times10^{-22}\), the \(2\times10^{-22}\) is unrepresentable in fp32, the function returns exactly 50.0, and hence \(\mathrm{logsigmoid}(-50)=-50.0\) bitwise-consistent with F.logsigmoid.

## Key points and common mistakes

1. **Never log(sigmoid(x)) for logsigmoid**: the −inf probe catches it; use \(-\mathrm{softplus}(-x)\), and softplus must carry its linear-segment threshold (return z for z>20).
2. **celu needs expm1 + a float64 intermediate**: naive fp32 at α=1e6 either zeroes the whole negative branch (small |x|, \(e^z-1\) rounds to 0) or cancels away precision that ×α then amplifies (larger |x|) — the α=1e6 case hits both pathologies.
3. **Ignore semantics**: softsign ignores min/max/alpha, hardtanh ignores alpha, celu ignores min/max; implementing "ignore" as "validate/use" fails the non-default-parameter cross-checks.
4. **hardtanh's saturation is exactly the boundary**: clamp performs no rounding; do not embellish with approximate saturation.
5. **softsign is strictly open-interval**: every finite x has |y|<1; "saturates to ±1" (the tanh reading) is wrong.
6. **Mode dispatch and error surface**: unknown mode raises ValueError; dtype/shape preserved elementwise (float64 consistency at 1e-9).
7. **Each formula has a neighboring shadow**: softsign ↔ tanh (rational), hardtanh ↔ ReLU (bounded), celu ↔ ELU (alpha into the denominator), logsigmoid ↔ softplus (log side) — mixing up the formulas is the most basic and most common failure.

## Differences from neighboring problems

- **vs `relu` / `leaky-relu`**: ReLU's negative branch is 0 / a linear slope, unbounded below; celu's negative branch is bounded below by \(-\alpha\) and joins smoothly (C¹) at the origin; hardtanh is a "bounded ReLU" (slope 1 in the middle band, hard saturation at both ends).
- **vs `elu` / `selu`**: ELU's negative branch is \(\alpha(e^x-1)\) — alpha outside the exponential, the branch never straightens with alpha; celu's \(\alpha(e^{x/\alpha}-1)\) puts alpha in the denominator, so the branch tends to linear at large alpha (the identity limit), a property ELU lacks. selu additionally carries a fixed \((\alpha,\lambda)\) self-normalizing scaling.
- **vs `sigmoid`**: logsigmoid is the "log-side" readout of sigmoid (the numerical foundation of BCE); sigmoid itself is not in this problem, but the stable form exists precisely so that \(\log\sigma(x)\) never underflows.
- **vs `mish` / `swish` / `hardsigmoid` / `hardswish`**: the same smooth/piecewise family — swish gates with sigmoid, mish gates with softplus (\(x\cdot\tanh(\mathrm{softplus}(x))\), the same softplus stable form as here), hardsigmoid/hardswish are piecewise-linear approximations (the "hard" family alongside hardtanh). This problem's four fill in the remaining staples of torch's activation list.

## Cross-check convention and complexity

- Main criterion: direct allclose against the corresponding `torch.nn.functional` operators (softsign/hardtanh/celu/logsigmoid; fp32 atol=1e-6, fp64 1e-9), plus structural requirements on the stable forms (softplus threshold 20, expm1+float64); example values and extreme probes (\(\pm10^4\), α=1e6, x=−120) are measured backfills.
- Discriminating probes: the parameter-ignore triad, hardtanh saturation via torch.equal, celu's identity limit (α=1e6, atol=1e-4), logsigmoid finiteness plus the counter-assertion that the naive form really is non-finite, and the illegal-mode ValueError.
- Complexity: \(O(\mathrm{numel}(x))\) time (each mode is a single elementwise pass; celu adds one float64 cast, 2× intermediate memory); extra space is same-order temporaries.
