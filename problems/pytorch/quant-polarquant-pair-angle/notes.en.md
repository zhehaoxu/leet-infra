# Solution notes · PolarQuant polar pair quantization

## One-liner

Fold the post-RoPE key along "front half / back half" into $D_h$ 2-D vectors and switch to $(\phi, r)$; quantise the angle and the radius independently with floor-based min-max over token groups, pack both codes into one integer `indices = (rho << tbits) + theta`, and on reconstruction take the bin centre of each code before spreading them back to $D$ channels with $\cos/\sin$.

## Why "half with half", not adjacent channels

The pairing convention comes straight from how RoPE is implemented. `rotate_half` splits head_dim into two halves and concatenates: $\text{rot}(x) = [\,-x_{D_h:},\ x_{:D_h}\,]$, i.e. the rotation acts inside the plane spanned by $(x_d,\ x_{D_h+d})$. Upstream's `key_states.view(B, Nk, Nb, G, 2, D)` puts that `2` exactly between the two halves, and `phi = atan2(x[..., 1, :], x[..., 0, :])` takes the **second half as $y$ and the first half as $x$**. Folding adjacent channels $(x_{2d}, x_{2d+1})$ instead amounts to changing the rotation basis, so $\phi$, $\theta$ and `recon` are all wrong — `check_pairing_is_halves_not_interleaved` uses the pair $(2,5)$ precisely to catch that divergence.

## Denominator $2^{\text{bits}}$, floor, clamp: three things that must be read together

Upstream writes `floor((phi - tmn) / tscale)` with `tscale = (tmx - tmn) / 2**tbits`, so the maximum element's quotient is exactly $2^{\text{bits}}$, gets cast by `.to(uint8)` and is only pushed back to the top code by `clamp(0, 2**bits-1)`. "The maximum is representable" is therefore a property of the **clamp**, not of the denominator; using $2^{\text{bits}}-1$ (the usual affine-quantisation habit) leaves the maximum on $2^{\text{bits}}-2$ and shifts the whole group by one code. `check_max_lands_on_top_code_and_centre_is_inside_bin` sweeps `tbits ∈ {2,3,5}` and pins all three facts at once: the top element hits the ceiling, the minimum is code $0$, and the reconstructed angle lies strictly inside the group's span.

Reconstruction using $(\text{code}+0.5)\cdot\text{scale}+\min$ is spelled out in the authors' reference kernel: it first builds `pphi = tscale * arange(2**tbits) + 0.5*tscale + tmn` and then gathers, so it is the bin centre rather than the lower edge. That is also why polar quantisation never reconstructs the largest angle back to $\max$ — it reconstructs it to $\max - 0.5\cdot\text{scale}$.

## The negative wrap changes the binning

`atan2` returns $(-\pi,\pi]$ and `where(phi < 0, phi + 2π, phi)` moves it to $[0,2\pi)$. This is not cosmetic: after wrapping, the group minimum can be a different token's angle and every $\theta$ shifts. In `check_negative_angles_wrap_and_change_the_min`, $2\pi-0.1$ wraps to $\approx 6.18$, the group minimum becomes the token at $0.0$, and the code sequence turns into $[1,3,0,2]$; the same test confirms `atan2(0, 3) = 0` is not mistakenly wrapped to $2\pi$.

## Four choices that are this exercise's own contract (stated in the problem)

- Upstream casts the codes with `.to(torch.uint8)` before clamping, which silently truncates beyond 8 bits; this problem keeps int64 throughout with an explicit `clamp(0, 2**bits-1)`, so `rbits/tbits` can safely go to 6 or 7.
- Upstream divides by `tscale`/`rscale` unconditionally, so a constant group yields $0/0=$ NaN; this problem defines `where(scale > 0, floor(...), 0)` and gives every entry of a constant group code $0$. Since the bin-centre formula degenerates to $(0+0.5)\cdot 0 + \min = \min$ when $\text{scale}=0$, a constant group reconstructs **exactly**, and `check_constant_group_reconstructs_exactly` asserts `recon == keys` to $10^{-12}$.
- Upstream hard-codes `rbits=tbits=4` and `group_size=128` as class attributes; this problem promotes them to arguments, with $r_{\text{bits}}$ governing only the radius and $t_{\text{bits}}$ only the angle and the packing shift.
- Upstream packs into uint8; `indices` here is int64 but is still exactly `(rho << tbits) + theta`, hence bounded by $2^{\text{rbits}+\text{tbits}}-1$ (an invariant `check_random_grid_and_dtype_against_oracle` verifies).

## Grading points

Exactly 6 keys; `phi/radii/theta/rho/indices` have shape $(B,H,L/G,G,D_h)$ and `recon` has shape $(B,H,L,D)$; the three code keys are int64 while the two geometric keys and `recon` follow the input dtype. Internals are float64, so a float32 and a float64 input produce **identical codes** (`check_random_grid_and_dtype_against_oracle` compares `theta.tolist()` directly) and only `recon` differs by the float32 tolerance. Inputs are read-only.

## Edges

$D=2$ (so $D_h=1$) and `rbits=tbits=1` are legal, and `group_size = L` gives a single group. A non-multiple $L$, an odd $D$, a non-4-D `keys`, a dtype other than float32/float64, or a `bool`/float/non-positive `group_size`/`rbits`/`tbits` all raise `ValueError` — note that $L=0$ is also illegal (the test feeds `torch.zeros(1,1,0,4)`), so do not wave it through on the grounds that "$0$ is a multiple of every positive integer".

## Complexity

$O(B\cdot H\cdot L\cdot D_h)$ trig and comparisons; the per-group min/max and the code search are linear within a group, and packing is two integer bit operations, with no $2^{\text{bits}}$ codebook table materialised.

## Provenance

The `PolarQuant` repository (the authors' own implementation), line by line:
- `models/modeling_llama_polar.py` L135-157, `LlamaPolarGroupAttention.quantize_and_pack_nbit` (class attributes `rbits=4, tbits=4, group_size=128, residual_length=128`; decoding goes through the triton kernel);
- `models/kernel4group.py` L113-158, the block in `__main__` the authors label "Code Below is the pytorch implementation of our kernel" — the `atan2` and negative wrap, the $2^{\text{bits}}$ denominators, `floor + clamp`, the $+0.5$ centre in `pphi/rradii`, and `indices = (rho << tbits) + theta` are all there.

The repository README states only NeurIPS 2025 and **gives no arXiv identifier**, so this problem cites the code paths as the evidence; the 2502.02617 circulating in awesome lists was not verified against the repository and is not used as a source.

## Related problems

`quant-kivi-kv-asym` (mixed per-channel/per-token asymmetric KV quantisation) and `quant-vqllm-residual-vq` (per-token scale normalisation plus residual codebooks) also compress the KV cache, but both quantise scalar channels; this problem changes the **representation coordinate system** (angle × radius) and packs two codes into one word. `kv-triattention-phase-score` likewise recovers the phase of post-RoPE keys, but to score eviction rather than to quantise.
