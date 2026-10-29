# Notes: One Segment Maximum Fixes Overflow and Underflow at Once — and the Built-in Norm Will Not Scale for You

## Origin

This is not an invention but a decades-old library convention: netlib BLAS `DNRM2` has been a two-pass "scan the largest magnitude, scale, then accumulate squares" routine since the 1988 edition (that odd pair of `xtrap` arguments in the source exists precisely for it); C99 asks `hypot` not to overflow or underflow through the intermediate square, and glibc achieves it with bit classification — so `math.hypot(3e154, 4e154)` returns \(5\times10^{154}\) and `math.hypot(3e-200, 4e-200)` returns \(5\times10^{-200}\), while `torch.linalg.norm` and `numpy.linalg.norm` both answer `inf` and `0.0` on those two inputs. In deep learning the places that trip over it are concrete: gradient clipping, weight-norm monitoring, division guards under mixed precision — magnitudes far more extreme than activations, where an implementation that squares directly in float16 turns "should I clip?" into "it is already inf".

## Why Scaling Is Safe in Both Directions

\(|x_i|/m\in[0,1]\), hence \((|x_i|/m)^p\in[0,1]\): no term can overflow, and the term that attains the maximum is **exactly \(1.0\)**, so the segment sum is at least \(1.0\) and can never collapse to \(0\). That is a stronger guarantee than "avoid inf" — the failure at \(10^{-200}\) is not a loss of precision but a nonzero answer turning into \(0\), after which every `1/norm` blows up. Multiplying \(m\) back at the end merely changes the scale of a quantity that is already well posed.

The \(p=\infty\) case is the degenerate form of the same path: the scale factor is the answer itself and a power of a maximum is meaningless. For \(0<p<1\) the result no longer satisfies the triangle inequality (a quasi-norm), but neither the identity nor the algorithm cares, which is why \(p=0.5\) is allowed here.

## Why the Five Degenerate Cases Need a Precedence

The order is **NaN > inf > all-zero > empty > regular**, and each entry corresponds to a concrete failure mode:

- NaN outranks inf, which agrees with what `torch.linalg.norm` actually does (measured: a segment holding both NaN and \(\pm\infty\) yields NaN); what this problem adds is that the precedence must be written as an explicit test instead of being allowed to "emerge" along the scaling chain — once \(m\) is NaN, a slightly different order of `where` branches makes the same segment flip between NaN and \(+\infty\). Test with `isnan().sum() > 0`; do not rely on arithmetic contamination.
- \(\pm\infty\) anywhere gives \(+\infty\): a norm ignores sign, and \(-\infty\) is just as infinite.
- \(m=0\) needs its own \(0.0\) branch: otherwise \(|x_i|/m\) is all \(0/0=\) NaN and a perfectly zero vector comes back NaN — the single easiest line to get wrong here.
- An empty segment returns \(0.0\) rather than raising, via an explicit branch — **not** because reduction is naturally safe: `torch.zeros(0).amax()` raises RuntimeError outright, so the length check has to come first.

## Core Ideas and Pitfalls

- **`amax` propagates NaN**, so "take the max first, then branch" is a legitimate way to divert rows, but the max you get is NaN rather than a real maximum — never reuse it for scaling.
- **Do not manufacture NaN with `INF * 0.0`**: readers take it for a typo; write a `float('nan')` constant or use `torch.full_like`.
- **Intermediate math must be float64**: in float32 the square of \(2\times10^{38}\) is inf on the spot, and that is exactly where `torch.linalg.norm` dies; scaling pushes every term into \([0,1]\), so as long as the segment maximum itself is inside float32 range the result is safe — measured, `[2e38, 2e38]` in float32 gives the finite \(2.8284\times10^{38}\).
- **Letting a float32 result overflow to inf is acceptable**: `[3e38, 3e38]` as float32 outputs inf, and the judge only insists it must not be NaN.
- **`keepdim` with `dim=None`**: full reduction returns a 0-D tensor and `keepdim` is then irrelevant; when the reduced axis has length 0 the output shape comes from the remaining axes and every value is \(0\).
- **Signed zeros**: the norm of `[-0.0, 0.0]` is \(0.0\); taking absolute values removes the sign, so no special case is needed.

## How It Differs from Neighboring Problems

`reduce-axis` is about shapes and broadcast rules in axis reduction, not about numeric scale. `quant-blockfp-tile` and `quant-mx-microscaling` also take a per-block maximum as a shared exponent, but that is a quantization scale, not a reduction. `logsumexp` applies the same "subtract the maximum" idea to overflow of `exp`; here the target is overflow *and underflow* of a power. `nu-welford-online-variance` fights rounding drift from summation order, which is a precision issue rather than a representability issue. The distinctive angles of this problem are "underflow is equally wrong" and the explicit precedence over five degenerate values.

## Cross-Check Basis and Complexity

The oracle never scales: `Decimal` with 60 digits of precision accumulates \(\sum|x_i|^p\) directly and then takes the \(p\)-th root (both power and root computed in Decimal, with no float64 round-off in between), and `movedim` moves the reduced axis to the back so segments are fed one at a time — a path with nothing in common with the reference solution's tensor scaling. Comparison is relative, NaN asserted with `isnan`, infinities short-circuited with `==`. Cases cover \(p\in\{0.5,1,2,3,8,\infty\}\), random data at both \(10^{150}\) and \(10^{-150}\), mixed-magnitude segments, every dim/keepdim combination, the five degenerate values, empty axes, extreme float32 inputs and invalid arguments. Complexity is \(O(\text{numel})\): one `abs`, one `amax`, one elementwise division and one powered sum.
