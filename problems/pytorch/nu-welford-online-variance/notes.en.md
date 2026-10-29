# Notes: Why \(E[x^2]-E[x]^2\) Returns Exactly Zero Under a Large Offset

## Origin

Welford gave this element-by-element update in a 1962 internal report; in *The Art of Computer Programming* Volume 2 Knuth names the accumulate-sums-then-subtract method "the textbook algorithm" and analyzes its cancellation, crediting the incremental alternative to Welford. NumPy's `np.var` uses a two-pass method, while TensorFlow's `tf.nn.moments` and the summaries of streaming monitoring systems implement Welford or its parallel-merge variant. The interview settings are of three kinds: writing a fused BatchNorm statistics kernel, explaining "why the variance becomes 0 when the data is offset by \(10^9\)", and merging two workers' \((n,\bar x,M_2)\) triples (the Chan-Golub-LeVeque step). This problem covers only the element-wise half; the merge is its natural extension.

## What the Three-Accumulator Update Computes

After \(k\) elements, \(\bar x_k\) is the mean and \(M_{2,k}=\sum_{i\le k}(x_i-\bar x_k)^2\). When \(x_{k+1}\) arrives,
$$\bar x_{k+1}=\bar x_k+\frac{d}{k+1},\qquad d=x_{k+1}-\bar x_k$$
and the increment of the second central moment can be written as
$$M_{2,k+1}=M_{2,k}+d\cdot\big(x_{k+1}-\bar x_{k+1}\big)$$
that is, "distance from the old mean" times "distance from the new mean" — which is exactly why `M2 += d * (v - mean)` must come **after** `mean += d / n`: the two differences in that one line are taken against different means. Geometrically \(d\,(x_{k+1}-\bar x_{k+1})\ge 0\) always, so \(M_2\) is monotone non-decreasing and can never be driven to a small negative number by cancellation (the sum-of-squares form does sometimes hand out a "negative variance" of order \(-10^{-8}\)).

## Why the Sum-of-Squares Form Collapses to Exactly Zero

For \(x_k=10^9+k\) with \(k=1..4\): \(\sum x_k^2 = 4\times10^{18}+2\times10^{10}+30\), but ulp\((4\times10^{18})=2^{62-52}=1024\), so that \(+30\) does not exist; \((\sum x)^2/4 = 4\times10^{18}+2\times10^{10}+25\) rounds onto **the same double**. Subtracting gives 0, hence a variance of exactly 0.0 — not "slightly inaccurate" but information fully erased. A relative error of \(10^{18}\) would need 60 significant decimal digits to recover, and double has 16.

## Core Ideas and Pitfalls

- **`M2 += d * d`**: \(d\) is the difference against the *old* mean, so squaring it directly **overestimates** — on the example \([1,2,3,4]\) it returns 2.75 instead of \(5/3\).
- **Swapping the order**: moving `mean += d / n` first makes both differences relative to the *new* mean and **underestimates** — the same data gives 1.1666666666666667. Both directions are visibly wrong on the example, which is what pins them down.
- **Element-wise, not chunk-wise**: summing a chunk and folding it in as one element is wrong; merging chunks needs the Chan-Golub-LeVeque \(\delta^2 mn/(m+n)\) form, which this problem does not ask for.
- **The two \(n=1\) conventions**: the unbiased divisor is \(n-1=0\), specified to return NaN, while the biased one returns 0.0. The tests assert both separately; answering 0 or NaN for both fails.
- **The float32 trap**: `torch.tensor([1e9+1])` is float32 by default, which crushes \(10^9+1..4\) into a single 1e9 so even the true variance becomes 0. Test data must pass `dtype=torch.float64` explicitly, and competitors must widen before accumulating.
- **Non-finite values**: an inf entering \(d/n\) turns the mean into inf or NaN, so the rule is "if any non-finite value appears, return (nan, nan) for the whole stream" without distinguishing signs.

## How It Differs from Neighboring Problems

`nu-kahan-compensated-sum` is about "how to add a known list accurately", this one about "what to keep once you have added"; in `batchnorm-forward` the running stats are merged as an exponential moving average (\(\bar x_{new}=(1-m)\bar x_{old}+m\,\bar x_{batch}\)), which is neither single-pass nor exact; `ag-batchnorm-backward` is about the variance being counted twice in the backward graph. The exclusive point here is the hand-computable failure mode "the sum-of-squares form returns exactly 0.0".

## Cross-Check Basis and Complexity

The oracle is a two-pass method: `math.fsum` for the mean, then accumulate \((x-\bar x)^2\), a path unrelated to the Welford recurrence. Well-conditioned random data at \(n=2/33/101/1000\) must match within 1e-9 relative error, and the offset-\(10^9\) case additionally asserts that the sum-of-squares form deviates from the truth by more than 1e-3. The unbiased variance of 1000 consecutive integers is checked by hand against \(n(n+1)/12 = 83416.66666666667\). Time \(O(N)\) in one pass, state \(O(1)\).
