# Deep Dive: SEG Query Smoothing

## Where it comes from

*Smoothed Energy Guidance* (arXiv:2408.00760). The implementation follows HuggingFace diffusers' `src/diffusers/hooks/smoothed_energy_guidance_utils.py`: `SmoothedEnergyGuidanceHook.post_forward` first fixes the kernel length with `kernel_size = ceil(6*sigma) + 1 - ceil(6*sigma) % 2`, then calls `_gaussian_blur_2d`; the latter splits on `is_inf = sigma > sigma_threshold_inf` into a "convolution" and a "mean" path, takes the first `n*n` tokens, `permute`s them to `(B, D, n, n)`, uses `F.pad(..., mode="reflect")` plus `F.conv2d(groups=D)` on the convolution path, and finally writes the slice back. The guider-side four-branch combination (`forward` with `pred_cond_seg`) is byte-for-byte the skeleton of `diff-pag-dual-guidance` (order 504), so this problem does not re-issue it and only covers the genuinely SEG-specific piece.

## Key ideas and pitfalls

1. **"Infinite blur" is not plain uniform averaging**: for `sigma > threshold_inf` the code still **convolves**; the kernel is clamped to the grid scale and a huge \(\sigma\) makes it nearly constant, so the result is close to — but not equal to — the global mean, because reflect padding counts edge neighbours repeatedly. On a 3×3 grid the example yields 2.666667, not 3.0. Implementing this path as `patch.mean()` is caught bit-wise.
2. **The kernel length is made odd first, then clamped**: `ceil(6σ)+1−ceil(6σ)%2` maps 6σ to the smallest odd number not below it; then `min(k0, n − (n%2 − 1))`, whose ceiling is \(n+1\) for even \(n\) and \(n\) for odd \(n\) — not simply \(n\). Writing `min(k0, n)` loses one usable length when \(n\) is even.
3. **Padding is `reflect`**: only `reflect` avoids repeating the edge value; `replicate` and zero padding both change the border rows. The criterion is `F.pad`'s `mode`, not "looks about right".
4. **The pad width is `k // 2`**: symmetric because \(k\) is odd; the even-\(k\) case never arises here (upstream already forced odd).
5. **Reshape order**: `permute(0, 2, 1)` then `reshape(B, D, n, n)`, and on the way back `reshape(B, D, n*n).permute(0, 2, 1)`. Dropping one permute mixes channels with tokens; the shapes still line up but every value is wrong.
6. **Tail tokens are untouched**: when `S` is not a perfect square, the last \(S-n^2\) tokens belong to no cell and must keep their exact values.
7. **Grouped convolution arity**: `groups=D` with a `(D, 1, k, k)` kernel convolves each channel independently; `groups=1` with a single-channel kernel silently mixes channels.

## How the judge cross-checks

The reference solution computes in float64 and returns float32. The test `_oracle` avoids `F.pad` / `F.conv2d`: reflect indices are folded back with a hand-written python `while`, the convolution is a four-level scalar loop, the Gaussian kernel is rebuilt from `math.exp`, and the kernel length is recomputed by a separate `_kernel_size`. Eight assertion groups: both example branches; the mean branch (per-(b, d) mean, differing across channels); threshold strictness (\(\sigma=\text{thr}\) takes the mean path, \(\text{thr}<\sigma\) the convolution path); kernel clamping (\(n=6,\sigma=1.5\) gives \(k=7\), cross-checked against the unclamped \(k=9\); the ceiling is 3 for \(n=3\)); tail-token preservation (including \(S=1\)); six (n, D, σ, thr) combinations plus a non-square token count; no in-place mutation; and invalid inputs. Tolerance `atol=1e-5`.

## Complexity

Reshape and write-back move \(O(BDn^2)\) elements; the convolution costs \(O(BDn^2k^2)\) multiply-adds with \(k\le n+1\), i.e. \(O(BDn^4)\) worst case. Kernel construction is \(O(k)\).
