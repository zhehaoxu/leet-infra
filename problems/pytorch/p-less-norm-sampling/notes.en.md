# Solution Notes · P-less-norm Sampling

## The only difference from P-less: threshold normalization

The original P-less sets `thr = Σp²` directly. P-less-norm applies an affine transform:

$$\text{thr}_{\text{norm}} = \frac{v\,\Sigma p^2 - 1}{v-1}$$

mapping the domain of $\Sigma p^2$, $[1/v,\,1]$, linearly onto $[0,\,1]$:

| Distribution shape | $\Sigma p^2$ | P-less thr | P-less-norm thr |
|----------|-------------|-----------|-----------------|
| Uniform ($1/v$) | $1/v$ | $1/v$ | $0$ |
| Peaked ($p_{\max}\to1$) | $\to 1$ | $\to 1$ | $\to 1$ |

The two coincide at the peaked end, but in the middle **the norm threshold is lower** → more tokens satisfy $p_i\ge\text{thr}$ →
the kept set is larger and sampling is more divergent. This is exactly the meaning of "relaxed from p-less, favor diversity" in the paper.

## Validity (it is never emptied)

We need to prove $p_{\max}\ge\text{thr}_{\text{norm}}$ always holds: from $\Sigma p^2\le p_{\max}$,

$$\text{thr}_{\text{norm}} = \frac{v\,\Sigma p^2 - 1}{v-1} \le \frac{v\,p_{\max}-1}{v-1} \le p_{\max}$$

The second inequality is equivalent to $v\,p_{\max}-1 \le (v-1)p_{\max} \iff p_{\max}\le 1$, which always holds. Hence the argmax
always satisfies the threshold, and the `filtered` denominator is always positive.

## Implementation notes

`v = probs.shape[-1]`; `thr = (v*probs.pow(2).sum(-1,keepdim=True) - 1)/(v - 1)`;
`keep = probs >= thr`; after normalization take the first $\ge u$ on the CDF of `filtered` in its original order. The official implementation uses
`mask = probs < p` to remove those **strictly smaller**, which is equivalent to keeping `probs >= thr` (this problem uses `>=`).
Note `V ≥ 2`, otherwise the denominator is 0.
