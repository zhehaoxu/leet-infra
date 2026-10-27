# In Depth: LpPool and maxpool Indices/unpool

## Source

- Operators being replicated: `torch.nn.functional.lp_pool2d` / `lp_pool1d` (the functional form of `nn.LPPool2d`), `F.max_pool2d(..., return_indices=True)`, `F.max_unpool2d` — PyTorch official documentation plus measurements on this repository's torch 2.14.1 CPU build.
- **How p-norm pooling relates to max pooling**: as \(p\to\infty\), \(\big(\sum|x_i|^p\big)^{1/p}\) converges to \(\max|x_i|\), which is why LpPool is often used as a smoothed max-pool (gradient spreads over the whole window instead of only through the argmax). The task limits \(p\) to finite values \(\ge 1\) and hands the \(\infty\) end to the `maxpool_idx` mode.
- **"Scatter the maximum back to its source position"** is the unpooling of Deconvnets (Zeiler & Fergus, *Adaptive Deconvolutional Networks for Mid and High Level Feature Engineering*, ICCV 2011, arXiv:1311.2901): record max positions (`return_indices`), then restore sparsely — exactly what `maxpool_idx` + `maxunpool` compose here.
- **Two places where the task deliberately departs from native** (both measured, see below): first, LpPool takes \(|x|\) of every element (native produces signed values or NaN for negative input when \(p\) is an odd integer or fractional); second, every invalid argument raises **ValueError** (native raises RuntimeError for `padding > kernel/2`, and `F.max_unpool2d` raises RuntimeError for an out-of-range index).
- `F.lp_pool2d` has **no padding argument**, so padded LpPool branches can only be checked against the independent oracle; `F.max_pool2d` does have padding and can be checked natively, indices included.

## Full derivation and index convention

**Window geometry**: per dimension \(o=\lfloor (n+2p_{\text{pad}}-k)/s\rfloor+1\) (the two spatial dimensions are independent). Window \(i\) covers padded coordinates \([is,\ is+k-1]\), which map back to input coordinates as \(r = is + a - p_{\text{pad}}\) with \(a\in[0,k)\). Both \(o\ge 1\) and \(2p_{\text{pad}}\le k\) are required; the latter guarantees **every window contains at least one real element** (the window spans \(k\), and at most \(k/2\) is added on each side, so the middle section must land inside the input). This matters most for `maxpool_idx`: a window made entirely of padding would emit \(-\infty\) and leave the index undefined.

**The index coordinate chain** (the single most error-prone line here): within window \((i,j)\) the \(k_h\times k_w\) block is flattened row-major into a local number \(l = a\,k_w + b\); `argmax` takes the first maximum (on CPU `torch.argmax` returns the lowest tied index, which is also what native `return_indices` does). Converting back to input coordinates:
\[
\text{row} = i\,s_h + \lfloor l/k_w\rfloor - p_{\text{pad},h},\qquad
\text{col} = j\,s_w + (l \bmod k_w) - p_{\text{pad},w},\qquad
\text{idx} = \text{row}\cdot W + \text{col}.\]
Two details: **subtract the padding offsets** (otherwise indices are expressed in the padded map and shift down/right whenever padding > 0) and **multiply by the original width \(W\)**, not the padded width \(W+2p_w\). The reference writes exactly `(rows - ph) * w + (cols - pw)` and, measured over the four configurations \((k,s,\text{pad})=((2,2),(2,2),(0,0)),((3,3),(1,1),(1,1)),((3,3),(2,1),(1,1)),((2,3),(1,2),(1,0))\), matches `F.max_pool2d(..., return_indices=True)` bitwise under `torch.equal`.

**Exactly where the \(|x|\) convention diverges from native**: assume a window contains negatives. For even \(p\), \(|x|^p = x^p\) and the two agree (measured identical for \(p=2,4,6\)); for odd integer and fractional \(p\) the difference is observable — window \([-1,-2,3,4]\):
- \(p=1\): this task gives \(1{+}2{+}3{+}4=10.0\), native `F.lp_pool2d(x,1,2,2)` gives **4.0** (signed sum);
- \(p=3\): this task gives \((1+8+27+64)^{1/3}=100^{1/3}=4.6415887\), native gives \(82^{1/3}=4.3444815\);
- \(p=1.5\): this task gives 6.6178598, native gives **NaN** (fractional power of a negative).
Conversely, with non-negative input even \(p=1\) and fractional \(p\) match native — the test suite uses exactly that contrast to draw the line between "our convention" and "native's".

**p=1 is not a mean**: `avg_pool2d(|x|, k)` returns \(\frac{1}{k^2}\sum|x_i|\) (judging asserts this task's output is exactly \(k^2\) times it). With `kernel_size=1` the window has one element, so the output is \(|x_i|\) elementwise — note this is an *exact* statement: in float32 \(p=2\) round-trips bitwise (measured `torch.equal` true), while \(p=3\) carries a \(1.2\times10^{-7}\) round-trip error from \(\text{pow}(p)\to\text{pow}(1/p)\), which is why judging uses `allclose` rather than `equal` there.

**Zero padding and all-zero windows**: LpPool pads with 0 and \((0)^{1/p}=0\), so an all-zero window (including padding-only corners) outputs 0 — the sum being 0 never turns into NaN under the root (\(0^{1/p}=0\) is well defined for \(p\ge1\)). This contrasts with maxpool's \(-\infty\) fill, and is precisely why the two modes need separate fill values.

**maxunpool as a scatter, and the injectivity condition**: the implementation is `zeros(N,C,H·W).scatter_(2, indices, x).view(N,C,H,W)` — "overwrite, everything else 0". `stride ≥ kernel` makes windows disjoint, so indices differ across windows (injective). Under overlap one source position could be hit by two windows and native `F.max_unpool2d` **does not define** which value wins; the task raises ValueError instead of guessing. With `output_size=None` the canvas is inferred as \(H=(h-1)s_h+k_h\), which is the inverse transform only in the no-leftover case; with leftovers (3×3 input, k=s=2 → 1×1 output) the inference yields 2×2 while the true input was 3×3, and index 4 falls out of range — that is exactly the worked example, which must pass `output_size=(3,3)`.

**The legal index range is the output canvas, not the small map**: validation uses \(0\le\text{idx}<H\cdot W\) where \((H,W)\) is the unpool output size. Native raises `RuntimeError: Found an invalid max index: 4 (output volumes are of size 2x2)` in the same situation; this task raises ValueError with an equivalent range message.

**Rank and argument checks**: `lppool1d` needs 3-D, the other modes 4-D; `lppool1d` additionally requires kernel/stride/padding to be equal across dimensions, because a 1-D tensor has one spatial dimension and a 2-tuple can only be its repetition. `maxpool_idx`/`maxunpool` ignore `p` entirely (judging asserts that passing `p=123.0` changes nothing), and an invalid `p` is therefore not checked there either.

## Key points and common mistakes

1. **Take \(|x|\) per element**: negative input with odd/fractional \(p\) is a deliberate **convention difference** from native, not a bug; judging cross-checks native on non-negative input and the oracle/probes on negative input.
2. **`p=1` sums \(|x|\)**, it is not a mean: dividing by an extra \(k^2\) (i.e. writing `avg_pool2d(|x|)`) is the most common error.
3. **Pad maxpool with \(-\infty\)** (the convention established by the `maxpool2d` problem): padding with 0 would make an all-negative window output 0.
4. **Subtract the padding offset in the index**: `(row - ph) * W + (col - pw)`; writing `row*W + col` misplaces every index as soon as padding > 0.
5. **The index stride is the original width \(W\)**, not the padded width.
6. **Ties take the row-major-first maximum**: flatten then `argmax`; recovering coordinates by comparing against `amax` with `==` can pick a different occurrence.
7. **Unpadded positions are 0** — not the original value, not \(-\infty\), not the other window elements.
8. **The unpool canvas size**: `output_size=None` only reconstructs the input when there are no leftovers; the worked example (3×3→1×1) must pass \((3,3)\) explicitly.
9. **Overlapping configurations are forbidden** (`stride < kernel` → ValueError): native leaves the write-back order undefined, so don't try to guess it with `scatter_reduce(amax)`.
10. **One error type: ValueError.** Don't copy native's RuntimeError/IndexError; rank, mode, \(p\), padding, indices and output_size all raise ValueError.
11. **Never mutate the input**: `unfold` returns views, so all arithmetic happens on the padded copy, and writing `out[p] = ...` requires cloning first (the reference only reads).

## Differences from neighboring problems

- **vs `maxpool2d`**: that problem returns only the maximum (and establishes the \(-\infty\) padding convention); this one adds the `return_indices` coordinate chain and the unpool inverse on top of it.
- **vs `avgpool2d`**: its difficulty is the mean plus the `count_include_pad` denominator game; here \(p=1\) deliberately does **not** divide by \(k^2\), and judging has a dedicated multiple probe.
- **vs `adaptive-maxpool` / `global-pooling`**: those derive window borders from a given output size (floor/ceil splits); this is classic fixed \(k/s/\text{pad}\) forward geometry.
- **vs `roi_pool`**: ROI pooling windows are floating-point coordinates landing anywhere; here windows always sit on the integer grid.
- **vs `unfold-fold-roll`**: that problem implements general im2col (with dilation) and `fold`'s **overlapping accumulation**; here `unfold` is only an internal device and `maxunpool` is an injective overwrite, with the overlapping case excluded by contract.
- **vs `scatter-reduce-aggregate` (same batch)**: unpool is exactly that operator's "overwrite" reduction with `include_self=False` and caller-supplied indices, in one dimension — and the duplicate-coordinate aggregation that problem studies is precisely what this one forbids.
- **vs `conv2d-transposed`**: transposed convolution also uses the write-back/accumulate duality, but each window value is a weighted sum rather than a moved argmax.

## Cross-check convention and complexity

- **Primary oracle is native**: `F.lp_pool2d` / `F.lp_pool1d` (with \(p=2\) and even \(p\), negative input included), \(p=1\) and fractional \(p\) on non-negative input, `F.max_pool2d(..., return_indices=True)` (output with \(10^{-6}\), indices with `torch.equal`), `F.max_unpool2d` with `torch.equal` (bitwise), and a float64 group at \(10^{-12}\).
- **Independent oracle: nested Python loops** (`_oracle_lppool2d`, `_oracle_lppool1d`, `_oracle_maxpool`, `_oracle_unpool`): they iterate \((n,c,i,j)\) window by window, pick the real value or \(-\infty\) by coordinate bounds, and accumulate pure-Python `**` powers. They cover what native cannot express (LpPool padding, odd/fractional \(p\) on negative input) and the negative-coordinate corners of the index chain.
- Discriminating cases: the three-mode example (\(\sqrt{46}\) / 5.0 / index 4 / a single non-zero unpool slot), the "sum not mean" probe for \(p=1\) against `avg_pool2d(|x|)`, elementwise \(|x|\) at kernel=1, all-zero windows and zero-padded corners, tied maxima (input rounded to thirds to create duplicates — measured: `torch.argmax` takes the lowest index, native `return_indices` agrees), `maxpool_idx` ignoring `p`, `stride=None → kernel`, the round-trip's non-zero mask matching the indices exactly (built with `scatter`), and 21 ValueError cases.
- Complexity: \(O(N\cdot C\cdot o_h\cdot o_w\cdot k_h k_w)\) time; the reference uses two `unfold` calls to get a \((N,C,o_h,o_w,k_h,k_w)\) view and reduces vectorised, so memory is the window-tensor expansion (irrelevant at judging sizes: \(H,W\le 8\), \(L\le 11\), \(k\le 4\)). maxunpool is an \(O(N C h w)\) scatter.
