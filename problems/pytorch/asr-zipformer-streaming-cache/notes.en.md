# Pitfalls (order 828)

## Sourcing notes

- Repo path: `k2-fsa/icefall`, `egs/librispeech/ASR/zipformer/zipformer.py` (2465 lines) plus `ChunkCausalDepthwiseConv1d` in `scaling.py`. The official doctest (`_test_zipformer_main`) only checks that forward runs; all numeric assertions here are self-built cross-checks.
- Per layer there are 6 cached tensors: `(cached_key, cached_nonlin_attn, cached_val1, cached_val2, cached_conv1, cached_conv2)`, laid out as `states[i*6:(i+1)*6]`.
- `chunkwise_conv` is **constructed with `padding=kernel_size // 2`** (symmetric conv pads itself); `streaming_forward` slices `x[..., left_pad:]` and convolves directly — omit that padding and the output loses k-1 frames (my first version used padding=0 and hit an immediate RuntimeError, caught by shape).

## Test-design pitfalls

1. **Broadcasting trap ((h,b,dp)*(h,dp) → (h,h,dp) when b=1)**: the manual cross-check `ph[:,:,t,:] * pos[0,rel,:].reshape(h,dp)` broadcasts the b axis into h when b=1, producing shape (h,h,seq,r) with misaligned values, failing the whole softmax comparison. Fix: reshape(h,1,dp) explicitly. Judge mirror: the solution uses matmul and never trips this, but the cross-check side must align explicitly.
2. **torch.stack axis slip**: the outer `stack(dim=-2)` on an (h,b,L+seq) tensor inserts the new axis at the b position → (h,seq,b,r); with b=1 the shape (h,3,1,5) differs numerically from the expected (h,1,3,5). Fix: explicit dim=2. Lesson: **when building 4-D tensors via list comprehension + stack, always pass the dim as an explicit number, never negative**.
3. **chunk_scale overlap**: when cs and k overlap (columns 3-4 for cs=8, k=5) the two corrections add — the first assertion `out[:, :5] == 1+scale[0]` is false on overlap columns. Cross-checks must split into three bands (left / overlap / right).
4. **Single-chunk semantics assert**: the official streaming_forward consumes exactly one chunk per call. My first draft let the mode accept seq=8 + chunk_size=4 at once, which made chunk_scale(cs=4) and x_chunk(seq=8) mutually exclusive in shape. Fix: `assert seq == chunk_size` pins single-chunk semantics; multi-chunk composition is delegated to stream_equiv.
5. **as_strided derivation**: the official stride transform `s2-s3` with `storage_offset=s3*(seq-1)` expands to `orig[h,b,t,(seq-1-t)+j]`. It is easy to flip the relative direction (t-j vs j-t); a single-element three-way comparison (manual dot / raw ps read / direct dot) pins it down.

## Mutation design (15 mutants, all CAUGHT)

- Cache family: take-head / no-update / shifted roll — covered by cache_roll and the conv-cache assertions together;
- Relative position: index without t — the per-element manual cross-check must stay (gather-internal mutants survive a softmax-only check);
- mask -1000→-1: masked-column weight < 1e-12 assertion (about 0.27 with -1);
- value_out cache drop: RuntimeError by shape plus the identity assertion;
- Gate family (nonlin/conv sigmoid dropped): "gate must participate" assertions (not allclose to the gate-free version at 1e-3);
- chunkwise padding loss: RuntimeError by shape;
- causal weights swapped for chunk_w: closed-form cross-check fails;
- scale left/right swap / missing 1.0 base: three-band assertions plus full-table comparison;
- whole identity break (causal shifted one frame): divisibility identity assertion fails.
