# Solution Notes · vqllm std Normalization + Multi-Step Residual Vector Quantization

## In one sentence

First divide each token's vector by its own unbiased standard deviation, entering a coordinates system where every token has the same scale; then run several quantizers in relay, each cutting the current residual into $N$ segments, taking the nearest codeword per segment in its own sub-codebook and subtracting it, running `num_residual_steps` rounds before handing over to the next quantizer; on reconstruction multiply the scale back and, on the key side, restore the interleaved channel order. This problem additionally requires recording every residual norm and every offset-carrying index along the way.

## Why divide by each token's own standard deviation

The codebook is static and shared by all tokens, while the assignment step does exactly one thing: find the codeword with the smallest Euclidean distance among $M$ candidates. That "nearest" is not neutral about absolute scale -- tokens whose amplitude is two orders of magnitude larger all land on the top end of the codebook and eat up the codewords, while small-amplitude tokens can only land on codewords that are equally too large, so their relative error explodes. After dividing by $\sigma_t=\operatorname{std}(x_t)$ every token is pulled back into the same scale band, and the codebook only has to cover the normalized manifold.

The scale information is not lost: it becomes one float per token (exactly the returned `scale`) and is multiplied back on reconstruction. So `error = ||final residual|| * scale` and the `residual_norm` trajectory are the same quantity read in two coordinate systems -- the trajectory is computed in normalized coordinates, `error` converts it back to the original unit.

Two details that are easy to change by reflex: `std` is **unbiased** (`correction=1`, denominator $D-1$), and upstream only divides, it **does not subtract the mean** (`x = x / scale`). The first affects the scale of every token; the second is why "a constant token's $z$ is not 0" (see below).

## The residual chain: $N$ independent segments, subtract per round, relay per quantizer

One quantization = $N$ mutually independent nearest-neighbour lookups: segment $k$ is compared only against the $M$ codewords of `codebooks[q][k]`, then the $N$ selected codewords are stitched back along the segment axis into $(B,L,D)$ and subtracted from the residual as a whole. The resulting residual goes into the next round (the same codebooks again); after $S$ rounds the next quantizer takes over, and the residual it receives is precisely what the previous one emitted -- hence `residual_norm[q][0] == residual_norm[q-1][-1]`.

The four degrees of freedom map cleanly onto four ways of spending bits: $\hat d$ is how many dimensions one codeword covers, $N$ is how many segments the vector is cut into, $M$ is how many candidates each segment has, and $S$ versus $Q$ are how many rounds one quantizer reuses itself versus how many quantizers relay. Both $S$ and $Q$ shave the same residual; they differ only in "reuse your own codebook" versus "switch codebook".

The index space is each quantizer's own $[0,N\cdot M)$: the result of segment $k$ gets $k\cdot M$ added. So collisions across different quantizers are **normal**, and $M$ may even differ per quantizer (measured: with $M=5$ in the second book the offsets are $0,5$ rather than $0,3$), and the bound the test applies is only `indices.max() < N * max(M)`.

## Channel reordering lets a sub-codebook see cross-channel coordinates

`view(B,L,N,dhat).transpose(-1,-2)` swaps the last two dimensions, so segment $k$ no longer gets "a contiguous run of $\hat d$ channels" but the cross-channel coordinates "the $k$-th channel of every sub-segment":

```
view (N=3, dhat=4)            transpose -> (dhat, N)         flattened w
c0 c1 c2 c3 | c4 c5 c6 c7 | c8 c9 c10 c11
w[k*N+m] = c[m*dhat+k]
w = c0 c4 c8 | c1 c5 c9 | c2 c6 c10 | c3 c7 c11
segment 0 (sub-codebook 0): c0 c4 c8
segment 1 (sub-codebook 1): c1 c5 c9
segment 2 (sub-codebook 2): c2 c6 c10
```

The motivation is the correlation structure: the $M$ codewords of one sub-codebook were designed for "the adjacent $\hat d$ dimensions", yet adjacent channels differ enormously in scale and correlation (a few feature channels carry most of the variance), so a contiguous run of channels typically mixes an "outlier channel" with ordinary ones and a single codeword would have to fit two magnitudes at once. After interleaving, one sub-codebook is responsible for evenly spaced coordinates spread over the whole vector, and sampling across channels makes each segment's distribution more homogeneous -- and closer to the distribution the codebook saw during training. Upstream does this on keys only (the value entry point simply delegates, with no reordering at all).

`split(dhat,-1) + stack(-1) + reshape` is its inverse. Restoring and multiplying by the scale commute: `scale` is $(B,L,1)$ and invariant under a channel permutation, so "multiply then restore" (upstream's order) and "restore then multiply" (the reference solution's order) give bitwise identical results.

## Five conventions that are easy to get wrong

1. **Unbiased, and no mean subtraction**. Writing `sqrt(mean(z^2))` makes `scale` systematically too small and `scale` and `quantized` go wrong together; conversely, anyone in the habit of "standardizing" easily writes `x - x.mean(-1)`, which is a different algorithm -- neither this problem nor upstream subtracts the mean. Measured with `x = [5,5,5,5]`: `scale` turns from 0 into 1, $z$ is still $[5,5,5,5]$, the reconstruction is `[0.5, 1.0, 1.5, 3.0]` and `error = 7.245688`.
2. **A zero scale is replaced by exactly 1**, not by `clamp_min(eps)`. Upstream writes `torch.where(scale == 0, 1, scale)`; substituting eps turns a constant token's $z$ into an astronomically large number and the whole chain blows up with it. What is guaranteed is "no NaN/Inf", not "the reconstruction is 0".
3. **Indices carry the $k\cdot M$ offset**, with $M$ taken from the current quantizer; converting back to a codebook entry number means subtracting $k\cdot M$.
4. **Ties go to the smaller entry number**. Upstream's `sim.argmax(dim=1)` leaves the tie order undefined; this problem pins it to the smallest index, and the tests deliberately create ties with $\pm$ symmetric entries and with bit-identical duplicate entries. The safe formulation is `hit = dist == dist.min()` followed by the first `True` in `hit` (which is exactly what numpy's `argmax` over a boolean mask does) -- do not rely on `argmin`'s tie behaviour.
5. **The splice points of the trajectory**: quantizer $q$ has $S+1$ values, where index 0 is the moment it **takes over** and the last one is the value **after all $S$ rounds**; `error` equals the last value of the last quantizer times `scale`, and `quantized` uses $z-\text{residual}$ from that same round.

## What is judged

- the shapes and dtypes of the five keys: `indices` is always int64, while the other four (`scale`/`residual_norm`/`quantized`/`error`) all follow the input dtype (float32 in, float32 out);
- offsets and trajectory: the first value of `residual_norm` is exactly $\lVert z\rVert$, the last value times `scale` is exactly `error`, and when `std == 0` the `scale` must be 1 with no NaN anywhere;
- the directed reorder assertions: with $N=2,\hat d=2$ the reorder turns $[c_0,c_1,c_2,c_3]$ into $[c_0,c_2,c_1,c_3]$ and the first segment becomes $(c_0,c_2)$; `indices` must change accordingly, `scale` must agree under both settings, and `quantized` must differ;
- ties take the smaller index, in both the $\pm$ equidistant case and the duplicate-codeword case;
- the multi-quantizer relay: `residual_norm[1][0] == residual_norm[0][-1]`, and the second quantizer must push the error down another order of magnitude;
- every invalid input raises `ValueError` (not 3-D, integer dtype, empty list, 2-D codebook, dtype mismatch, $N\hat d\ne D$, `steps=0`, `steps` as a bool, `reorder` given as 1, and so on);
- randomized comparison: many shapes x multiple rounds x multiple quantizers x both reorder settings, every key compared against an **independent numpy oracle** (direct squared differences, not the expansion);
- arguments are never modified in place: each case clones first, runs, then compares back with `torch.equal`.

## Boundaries

- $D\ge 2$, $N\ge 1$, $M\ge 1$ and $N\hat d=D$. $N=1$ is legal (a shared codebook; the $N$ segments degenerate to one); with $\hat d=1$ the reorder is the identity permutation.
- **The residual norm is not guaranteed to decrease per round.** Nearest-neighbour minimizes $\lVert s-c\rVert$, not "make the residual smaller"; when the codebook has no codeword near the origin, subtracting a codeword can grow the residual. Measured: an all-zero token with the example codebook of this problem gives the trajectory $[0,\,0.25]$. The monotonicity assertion in the tests only applies to a deliberately nested codebook.
- `scale` agreeing across the two reorder settings is a mathematical fact, but in floating point the reordering changes the summation order and a tail difference of about $10^{-16}$ can appear; the tests compare with `_close` tolerances (2e-9 for float64, 3e-5 for float32), not `torch.equal`, so do not special-case `scale` just to make it bitwise equal.
- The order of the validation checks is deliberate: first `x`'s rank and dtype, then `reorder_channel`/`num_residual_steps`, then $N\hat d=D$, and finally each codebook's shape and dtype one by one. So `D=1` is caught by the "$D\ge2$" rule, not by the codebook shape rule.
- `codebooks` accepts only list/tuple: a generator, and a "list containing plain lists", both raise `ValueError` (the latter at the shape check). A non-contiguous `x` or one with `requires_grad=True` computes fine -- everything is out-of-place plus `.detach()`, which also satisfies "arguments must not be modified".

## Complexity

Let $T=BL$. Per quantizer, per round, per token there are $N$ segments x $M$ codewords x $\hat d$ dimensions of comparisons, i.e. $O(MD)$, for a total of $O(T\,Q\,S\,M\,D)$; the expansion and the direct squared difference cost the same, the former just written as one batched matmul. The output is part of the bill too: `residual_norm` is $Q(S+1)T$ and `indices` is $T\,Q\,S\,N$ int64 values. On the compression side the bit accounting is $N\log_2 M$ bits of index per token per quantizer per round, plus one `scale` float per token -- codebook capacity $M$ and residual rounds $S$ trade against each other inside the same budget.

## Source

`iankur/vqllm` (arXiv 2410.15704, ENLSP-IV 2024; the repository's own `README.md:3-4` gives both citations, matching what the problem statement claims). Every file:function:line below was read line by line in the local `/tmp/qsrc/vqllm`.

- `vqllm/quantizer/utils.py:22-25` `quantize()`: L23 `scale = x.std(dim=-1, keepdim=True)`, L24 `scale = torch.where(scale == 0, 1, scale)`, L25 `x = x / scale` -- the exact three lines behind rule 1: unbiased, division only with no mean subtraction, zero replaced by 1.
- `utils.py:41-53` `quantize()`: L44 the outer `for quantizer in quantizers:`, L48 the inner `for _ in range(quantizer.num_residual_steps):`, L49-51 `xq_, indices, commitment_loss_ = quantizer(x, ...)`, L52 `x = x - xq_` -- the "subtract per round, relay per quantizer" of rule 3.
- `utils.py:55` `x = (x_orig - x.detach()).to(x_dtype)` and `utils.py:58` `x = x * scale` -- rule 5's `z - final residual`, then multiply the scale back. (Note `x_orig` is captured at L38, after the division at L25 already happened, so it is the normalized $z$ all along.)
- `utils.py:62-80` `quantize_key()`: L64 `reorder_channel = quantizers[0].reorder_channel`, L71-72 `x.view(*x_shape[:-1], -1, dim // codebook_entry_dim)` followed by `transpose(-1,-2).reshape(*x_shape)`, L77-78 `x.split(codebook_entry_dim, dim=-1)` followed by `torch.stack(x, dim=-1).reshape(*x_shape)` -- the reorder and restore of rule 2.
- `utils.py:83-85` `quantize_value()` delegates straight to `quantize` and never touches the reordering -- this confirms the statement that upstream applies it to keys only. The wiring agrees too: `vqllm/models/_modules.py:91-93` sends k through `quantize_key` and L98-100 sends v through `quantize_value` (k/v have shape `(b, s, n_kv, 1, head_dim)`, see L84-86, which is exactly "one vector per token"); `vqllm/models/_component_builders.py:112` sets `reorder_channel` on the key quantizer only, and L113-115 express "several quantizers" as `nn.ModuleList([copy.deepcopy(quantizer_key) for _ in range(num_residual_codebooks)])`.
- `vqllm/quantizer/quantize.py:141-182` `VQVAEQuantize.decode()`: L146-147 `z.view(-1, num_codebooks, codebook_entry_dim)` plus `permute(1, 2, 0)`, L150-152 the codebook `view(num_codebooks, num_codebook_entries, codebook_entry_dim)`, L167-171 `sim = -(z.pow(2).sum(dim=1, keepdim=True) + codebook.pow(2).sum(dim=2, keepdim=True) - 2 * torch.matmul(codebook, z))`, L173 `indices = sim.argmax(dim=1).permute(1, 0)`, L176-178 `offset = torch.arange(self.num_codebooks) * self.num_codebook_entries` then `indices += offset`, L181-182 `z_q = self.codebook(indices).view(input_shape)` -- the expanded distance, the per-segment nearest neighbour and the $k\cdot M$ offset behind rules 3 and 4.
- The calibration side (declared out of scope by the problem, references verified): `quantize.py:4` `from scipy.cluster.vq import kmeans2`, `quantize.py:188-209` `init_codebook()` which seeds the codewords with `kmeans2(..., minit="points")`, and `quantize.py:9-46` `EmbeddingEMA` whose `cluster_size_ema_update`/`embed_avg_ema_update`/`weight_update` perform the EMA update.
- There is also a triton fast path, `vqllm/quantizer/quantize_kernel.py:181-208`: the same per-quantizer per-round residual loop (L181-182), the same expanded Euclidean distance (L200-206), the same `idx += k * BLOCK_SIZE_C` offset (L208); but `quantize.py:69-70` states plainly that it only supports a shared codebook ($N=1$), and `utils.py:27-36` requires `use_fast_quantizer` to reach it. This problem follows the eager branch.

**Correction log for the statement**: the second bullet of the "Constraints" section used to say that for a constant token "after replacing the 0 with 1, $z$ is all zeros and the reconstruction is all zeros too". That disagreed both with the statement's own rule 1 (`z = x / scale`, no mean subtraction) and with upstream `utils.py:25`. Measured, `x=[5,5,5,5]` gives $z=[5,5,5,5]$, reconstruction `[0.5,1.0,1.5,3.0]` and `error = 7.245688`; only an all-zero token makes $z$ actually 0, and the reconstruction is still the selected codewords (with the example codebook, `[-0.05,-0.1,-0.1,-0.2]` and `error = 0.25`). What the bullet has to guarantee is that `scale` is exactly 1 and no NaN/Inf appears, and both language versions of the statement now say that; `test_cases.py` only pins `scale == 1.0`.

**Additions of this problem** (left undefined by upstream, or done differently, and pinned explicitly by the statement):

- **Tie order**: upstream's `argmax` at `quantize.py:173` defines no tie order; this problem pins it to "smallest entry number".
- **`indices` is this problem's artifact**: at `utils.py:49-53` the eager branch simply discards the `indices` it receives (they are only fed to the EMA during training), so upstream's inference path returns no indices at all; this problem stacks them into an int64 tensor of shape $(Q,S,B,L,N)$.
- **The `residual_norm` trajectory is this problem's instrumentation**: upstream never records the residual norm before/after each round; both the $(Q,S+1,B,L)$ shape and the "splice per quantizer" convention are the exercise's own.
- **`error` is a different quantity**: upstream's per-token quantity is `commitment_loss`, a per-channel MSE (`quantize.py:130-131`); this problem's `error` is $\lVert\text{residual}\rVert_2\cdot\sigma$, not an MSE, differing in unit by a factor of order $\sqrt D$.
- **The internal precision is float64**: upstream casts internally to `.float()` (`utils.py:38,42`, `quantize.py:165-166`); this problem computes in float64 per the statement and returns in the input dtype.
- **`num_residual_steps` becomes a global argument**: upstream reads it per quantizer (`utils.py:48` reads `quantizer.num_residual_steps`), so different quantizers could legitimately run different numbers of rounds; here one int governs all $Q$ of them.
- **`reorder_channel` becomes an explicit argument**: upstream reads `quantizers[0].reorder_channel` (`utils.py:64`), and that attribute is not even defined in `VQVAEQuantize.__init__` -- it is monkey-patched onto the key quantizer afterwards at `_component_builders.py:112`. Here no such attribute is needed, and the flag is no longer restricted to keys.
- **Codebooks are explicit arguments**: upstream keeps the codewords inside `EmbeddingEMA` as a flat $(N\cdot M,\hat d)$ weight produced by `kmeans2` plus an EMA; here each quantizer is handed as an explicit $(N,M,\hat d)$ tensor and all training is stripped away.
- **There is no mask**: upstream carries a `(B,L)` `mask` everywhere (`quantize`/`decode`/the kernel, e.g. `quantize.py:88-89`, `quantize_kernel.py:164`); here every token is quantized unconditionally.
- **Euclidean only**: upstream has a `cosine_distance` branch (`quantize.py:160-162`), but both entry points hard-code `cosine_distance = False` (`utils.py:63,84`), so the problem's Euclidean-only rule is faithful rather than a modification.
- **Validation and error types**: upstream performs no input validation at all, and $N\hat d\ne D$ surfaces as a RuntimeError from `view`; this problem pins $D\ge2$, dtype, non-empty list, shared $(N,\hat d)$, $M\ge1$ and the bool/int rules (a bool does not count as an int) all as `ValueError`. The "$D\ge2$" requirement in particular is purely a consequence of the unbiased `std` and is not something upstream asks for.
- **Training branches removed**: `training`, `copy_grad` and the straight-through estimator `z_q = z + (z_q - z).detach()` (`quantize.py:137-138`) are all out of scope; upstream's eager branch additionally wraps the whole loop in `torch.no_grad()`.
- **The shape convention is narrowed to 3-D**: upstream's signature is `(bsz, seq_len, *, dim)` with any number of extra middle dimensions such as heads (see the docstring at `quantize.py:104-105`); this problem fixes $(B,L,D)$ and folds the head axis into either $D$ or $L$.

## Related problems

- `quant-squeezellm-sensitive-kmeans`: the other half of the same chain -- it *learns* the grid with sensitivity-weighted Lloyd; here the codebook is an explicit argument and only the assignment plus the residual subtraction on the inference side are judged.
- `quant-kivi-kv-asym`: also "one scale per token", but the assignment is a uniform affine grid with a zero-point; here the assignment searches for the nearest codeword among several sub-codebooks, and indices must be stored on top of the scale.
- `quant-double-quant-scale`: also a two-level scheme, but its second level quantizes the **first level's scales** (the metadata itself); here the second level quantizes the **first level's residual** -- a completely different direction.
