# Notes · C3A's Block-Circular Convolution

## Where it comes from

- `huggingface/peft`, `src/peft/tuners/c3a/utils.py`: `get_circulant_fast(w)` expands an `(m, n, bs)` kernel tensor into a dense `(m*bs, n*bs)` delta, and `BlockCircularConvolution` is the custom-autograd twin that turns the circulant product into an elementwise product with two FFTs.
- `src/peft/tuners/c3a/layer.py`: `C3ALayer.get_delta_weight` returns `get_circulant_fast(kernel) / base_layer.weight.size(-1)` (i.e. divided by `in_features`), and `update_layer` pins the kernel shape to `(out_features // block_size, in_features // block_size, block_size)`; the unmerged path of `C3ALinear.forward` is `base(x) + BlockCircularConvolution(x, kernel) / x.size(-1)`. This problem mirrors both call sites line by line.
- Paper: Chen et al., *Parameter-Efficient Fine-Tuning via Circular Convolution*, arXiv:2407.19342 (C3A). The formulas above are taken from the source, which is the contract of record.

## Key ideas and pitfalls

**The in-block index is `(j - i) % bs`; the direction matters.** A circulant matrix is defined so each row shifts left relative to the one above, hence `C[i, j] = k[(j - i) % bs]` and row 0 is exactly the kernel. Writing `(i - j) % bs` gives the right-shift variant, which is the transpose of the block: at `bs = 2` the two differ only in the off-diagonal entries, at `bs > 2` the whole block is scrambled — this is the classic way to fail `check_orientation`.

**The divisor is `d_in`, not `bs`.** In peft the denominator comes from `weight.size(-1)`, the last dimension of the convolution input. `check_scaling_by_in_features` keeps the same kernel values and grows `d_in` from 4 to 8; the delta halves exactly, a distinction that is invisible if you normalize by `bs`. Conceptually it plays the role of LoRA's `lora_alpha / r`.

**The two kernel axes track output blocks and input blocks respectively.** The shape is `(out//bs, in//bs, bs)`, where axis 0 is the row-block index `m` and axis 1 is the column-block index `n`. Swapping them is only a transpose on square layers, and it is caught immediately by the forward `x @ deltaᵀ` and by `check_block_locality` (editing `kernel[1,0]` must move only block `(1,0)` of the delta).

**At `bs = 1` the circulant degenerates into an ordinary dense matrix.** Each block holds a single element and `(j - i) % 1` is always 0, so `delta = kernel[:, :, 0] / d_in` is a full `out × in` matrix — the parameter advantage is gone. At the other extreme, `bs = d_in = d_out` yields a single block whose row `o` is the kernel shifted right by `o`. One check pins each end.

**A constant kernel gives a constant matrix.** When all `bs` kernel values are equal, cyclic shifting changes nothing and every entry of `delta` equals `c / d_in`; each row then sums to `c`. This is the quickest way to sanity-check the normalizer.

**Why circulants are cheap.** A circulant is diagonalized by the DFT: `C = Fᴴ diag(fft(k)) F`, so one in-block product becomes elementwise and drops from `O(bs²)` to `O(bs log bs)` — precisely why peft uses `fft`/`ifft` plus `einsum` instead of assembling blocks. The judge asks for the explicit `delta`, so this materialization is required; still worth remembering that C3A saves parameters (`out × in / bs`) and compute (frequency-domain multiply) — two different things.

**peft forces the kernel to float32.** The source comment notes that FFT support for fp16/bf16 is incomplete, so kernels are created in fp32 and the forward runs in fp32. This problem asks for float64 internals and float32 returns, which is why the comparison against real peft agrees to about `1e-7` rather than bit for bit.

## How the judge cross-checks

- `_oracle` in `test_cases.py` recomputes `delta` with four numpy `for m, n, i, j` loops, taking the cyclic index via a hand-written `while src < 0: src += bs`; it shares no code with the vectorized `permute/reshape` reference. `check_against_oracle` covers six configurations: `1×1×1`, non-square layers, `bs = 1`, `bs = in/2`, with bias, and pure float32 inputs.
- Structural checks each guard one pitfall: `check_block_layout` requires exactly `bs` distinct values inside a block and zeros elsewhere; `check_orientation` uses `bs = 3` to separate `(j-i)` from `(i-j)`; `check_single_block_circulant` fixes the shift direction; `check_scaling_by_in_features` fixes the divisor; `check_block_locality` fixes axis semantics; `check_merge_equivalence` verifies `y = x @ (W0 + delta)ᵀ + b0` and that `delta` does not depend on `x`.
- While authoring, real peft was used as a second oracle: a `C3ALinear(nn.Linear(in, out), ...)` whose `c3a_kernel` is overwritten with the same random kernel was compared against `solution.solve` over six `(in, out, bs)` configurations, giving a max delta difference of `7.5e-9` and a max forward difference of `4.8e-7` (all of it from peft's fp32 FFT path). That probe is not part of the judged files.

## Complexity

Let `N = prod(x.shape[:-1])`. Materializing `delta` costs `O(d_out × d_in)` time and memory — an inherent price of the return value, whereas real C3A stores only `d_out × d_in / bs` parameters. The forward pass is two dense multiply-accumulates, `O(N × d_out × d_in)`. On the frequency-domain path each in-block product drops to `O(bs log bs)`, so both parameters and compute scale by `1/bs`. All arithmetic runs in float64 and only the returned tensors are cast to float32.
