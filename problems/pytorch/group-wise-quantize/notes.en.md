# Group-wise Quantization Explained

## Why LLM weights need grouping

The weights of one output channel are unevenly distributed along the input dimension K: some K segments contain large outliers while others are all small. Per-channel gives the whole channel a single scale, so the small-value segments get "diluted" by the large-value segments down to almost all zero grid points. INT8 per-channel is still acceptable, but **INT4 has only 8 positive grid points**, and per-channel accuracy collapses. Group-wise splits the channel into small groups along K (128/64), each with its own absmax-derived scale, which amounts to "locally adaptive resolution" and is the key to GPTQ/AWQ compressing LLMs to W4 while retaining accuracy. The cost: extra scales to store plus per-group switching during dequantization, and hardware must align to `group_size`.

## group_size and the short tail

When $n$ is not divisible by `group_size`, the last group has only $n \bmod \text{gs}$ elements. **Its absmax must be estimated over its true length** — padding with zeros for alignment would not affect absmax (0 is not a new extremum), but padding with "the maximum value" or reshaping incorrectly will corrupt it. The safest formulation is the slice `start:start+gs`, which naturally stops at the end of the row for the short tail.

## qmax follows the bit width

The number of positive grid points in symmetric quantization is $q_{max}=2^{\text{bits}-1}-1$: int8→127, int4→7; the $-128$ end of int8 is unused (keeping symmetry about 0, consistent with `symmetric-quantize`). When bits is a variable, do not hard-code 127.

## Relation to other quantization granularities

| Granularity | Number of scale groups ([out,in]) | Typical use |
|---|---|---|
| per-tensor | 1 | activations, simple deployment |
| per-channel | out | weight int8 (`per-channel-quantize`) |
| group-wise | out × ceil(in/gs) | weight int4 (GPTQ/AWQ) |

Finer granularity means higher accuracy and more metadata. This problem groups along the in dimension (the last dimension).

## Grading notes

- An all-zero group has absmax=0 → return 0 unchanged (division-by-zero trap).
- `_ref` uses `index_reduce_` to vectorize the per-group absmax, while the solution's per-group slice loop is a different derivation path; the two cross-check each other independently.
- `check_example` uses the dumped int8/int4 values; there are also short-tail, gs>n, 3-D, int4, and input-not-modified cases.
