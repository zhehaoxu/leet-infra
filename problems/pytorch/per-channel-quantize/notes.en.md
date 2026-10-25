# Per-Channel Quantization Explained

## Why per-tensor falls short on weights

Different output channels of a weight have wildly different magnitudes: one channel has an outlier with $|w|=50$, while another stays entirely within $|w|<0.5$. Per-tensor picks a global absmax=50 to set the scale, so for the small-value channel $q=\operatorname{round}(0.5/ (50/127))=\operatorname{round}(1.27)=1$ — the whole channel is left with only a few grid points and almost all information is lost. Per-channel lets each small channel set its scale from its own absmax=0.5 and recovers full resolution. This is why almost every PTQ/QAT scheme uses per-channel by default for **weights** (activations generally stay per-tensor, since their dynamic range varies with the input and a per-channel scale cannot be fixed offline).

## axis semantics and the reduction dims

"per-channel along axis $a$" = the channel is dimension $a$, and every **other** dimension is shared within a channel. For `[C, in]` with `axis=0`, absmax must be reduced along `dim=1` to get a scale of length `C`. For multiple dims (conv weight `[O, I, kh, kw]`, `axis=0`) reduce along `dim=(1,2,3)`. Writing `reduce_dims = everything except axis` is the most general form; do not treat axis as "the dimension that gets reduced away"——it is exactly the opposite.

Normalize a negative `axis` first with `axis % dim`. When broadcasting the scale, reshape its shape to `[1,…,C,…,1]` (only the axis dim is C).

## Division by zero: the oracle's hidden convention

An all-zero channel → absmax=0 → scale=0. A direct `x/0` yields NaN, but `torch.fake_quantize_per_channel_affine` **returns 0** for scale=0 channels. So scale=0 channels must be handled separately (the solution here uses a `safe_scale` that replaces 0 with 1 for the division, then multiplies by the true scale=0 so the output becomes zero). This is the easiest point to miss, and it is where results stop matching the oracle.

## fake-quant vs. real integer storage

Both this problem and `symmetric-quantize` are **fake-quant**: they still return floats after the round trip (quantize → dequantize), used in QAT to let gradients pass straight through, or in evaluation to simulate quantization error. Only at real deployment time is $q$ stored as int8 with just a per-channel scale vector (see `weight-only-int8-matmul`, where the scale takes part in inference).

## Related

- Upstream: `symmetric-quantize` (the per-tensor version; this problem is its channel-wise generalization).
- Downstream: `group-wise-quantize` (a finer block granularity than per-channel), `weight-only-int8-matmul` (how to use it once int8 is actually stored).
- Judging: `check_example` uses dumped values; `check_matches_oracle` compares against `torch.fake_quantize_per_channel_affine`; there are also all-zero-channel, 3-D conv, negative-axis, and float64 cases.
