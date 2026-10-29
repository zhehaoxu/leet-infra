# Welford's Online Variance: a Three-Accumulator Update Over a Stream of Chunks

## Problem Description

Data arrives chunk by chunk: dataloader batches, streaming-training shards, monitoring windows. To get the global mean and var along that stream, the textbook move is to accumulate \(\sum x\) and \(\sum x^2\) and finish with \(E[x^2]-E[x]^2\) — but as soon as the data carries a large offset (say every value is \(10^9+k\)), subtracting two nearly equal large numbers erases dispersion of exactly that size. This problem wants Welford's single-pass three-accumulator update \((n, \text{mean}, M_2)\), advanced element by element, materializing nothing between chunks. Return `(mean, var)` as two Python floats.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- `chunks` is a list/tuple of float32/float64 tensors, flattened row-major into a scalar sequence; empty chunks are allowed and everything accumulates in float64.
- For each arriving scalar \(v\) perform exactly this step:

```
n += 1
d = v - mean
mean += d / n
M2 += d * (v - mean)
```

- At the end, `unbiased=True` returns \(M_2/(n-1)\) (sample variance) and `unbiased=False` returns \(M_2/n\) (population variance).
- Lines 3 and 4 cannot be swapped: \(M_2\) must use the **updated** mean, and writing `d * d` is the classic mistake.
- No accumulating sums and sums of squares to finish with \(E[x^2]-E[x]^2\), and no materializing the data for a two-pass scan.
- An empty stream (\(n=0\)) returns `(nan, nan)`; with \(n=1\) the mean is that element, `unbiased=True` gives var=nan and `False` gives 0.0; a ±inf or NaN anywhere in the stream also returns `(nan, nan)`.
- A non-list/tuple `chunks`, a non-tensor element, a non-float dtype, or a non-bool `unbiased` all raise ValueError.

## Example

```
chunks = [1.0, 2.0] [] [3.0, 4.0]

unbiased=True  ->  mean = 2.5   var = 1.6666666666666667  (= 5/3)
unbiased=False ->  mean = 2.5   var = 1.25

offset = [1e9+1, 1e9+2, 1e9+3, 1e9+4]

welford           ->  mean = 1000000002.5   var = 1.6666666666666667
E[x^2] - E[x]^2   ->  var = 0.0
```

The dispersion of \(10^9+k\) is only 1, while \(\sum x^2\approx4\times10^{18}\) already has ulp 1024, so the 5 left after subtracting \((\sum x)^2/n\) rounds to 0: the sum-of-squares form returns **exactly 0.0** while Welford returns the true \(5/3\). The empty chunk in the middle changes nothing.

## Constraints

- One pass, \(O(1)\) extra state (three scalars); how the stream is chopped only moves the last bit, well inside the 1e-9 tolerance
- the oracle is a two-pass method (`math.fsum` for the mean, then the second central moment), with 1e-9 relative tolerance on well-conditioned data
- the unbiased sample variance of \(n\) consecutive integers is \(n(n+1)/12\): \(n=4\) gives \(5/3\), \(n=1000\) gives 83416.66666666667, handy for a hand check
