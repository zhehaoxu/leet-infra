# Quantization: DuQuant ZigZag Channel Permutation — Algorithm Notes

## 1. Outlier channels and block-wise quantization

Block-wise quantization shares one scale per block. If one channel has a far larger magnitude than the rest of its block, that channel inflates the scale and wastes effective bits for every channel in the block. DuQuant (arXiv:2406.01721) applies a "dual transformation": a block-wise orthogonal rotation spreads outliers out, and a channel permutation distributes large-magnitude channels evenly across blocks. This problem keeps only the data-independent, fully deterministic permutation.

The permutation does not change weight values, only column order; at inference the activation side must be permuted the same way, which is why upstream implements it as a joint transform of weights and inputs. This problem returns only the permutation and the permuted weights, not the activation-side transform.

## 2. Algorithm steps

Given `W` (`O×H`) and block size `B` (`H % B == 0`), let `G = H / B`:

1. Key `key_c = max_r |W[r, c]|`;
2. Stable-sort `(c, key_c)` by `key` descending;
3. Snake deal: a cursor starts at bucket 0, walks down to bucket `G-1`, turns around, walks back to 0, turns again, and so on, placing each sorted column into the current bucket;
4. Re-sort every bucket by `key` descending, stably;
5. Concatenate bucket indices into `perm`.

The global sort in step 2 ensures that adjacent magnitude ranks land in different buckets; the per-bucket sort in step 4 keeps each block ordered from large to small for downstream block quantization and packing. The snake (rather than a fixed `c mod G`) reverses direction every pass so a block does not always receive columns on the same "diagonal".

## 3. Why the maximum absolute value is the key

Block quantization error is dominated by the largest magnitude (it determines the scale), so `max_r |W[r,c]|` measures a column's damage, not the mean or a norm. Upstream `permutation_zigzag` literally uses `weight.abs().max(dim=0).values`; for a 2-D linear-layer weight this is the per-column maximum absolute value. `permutation_random` is its randomized control branch and is not implemented here.

## 4. Deterministic boundaries

- Equal keys keep ascending original index (stable sort); this rule must be fixed or the order of equal-magnitude columns is not judgeable.
- `B = 1`: `G = H`, one column per bucket, so the result is the global descending stable sort; `B = H`: a single bucket, same result.
- When `H` is not divisible by `B`, upstream silently overruns/drops columns; this problem raises `ValueError` instead.
- Upstream's block rotation uses random QR (`torch.randn` + `torch.linalg.qr` in `get_rot`) and a `Rot.pkl` that is not distributed with the repository, so it is not reproducible. This problem explicitly excludes rotation and keeps only the permutation.

## 5. Division of labor within the collection

- `quant-rptq-channel-reorder-group`: sorts by **activation span** ascending, cuts contiguous balanced groups, and quantizes inside each group;
- This problem: sorts by **weight magnitude** descending, deals columns into blocks in a **snake interleave**, re-sorts inside each block, and returns only the permutation without quantization;
- Hadamard/rotation family: `quarot-hadamard-quant` and `quant-hadamard-transform` already cover rotations, so DuQuant's rotation is not repeated;
- No overlap with codebook problems such as `quant-squeezellm-sensitive-kmeans`: this problem produces no codebook.

## 6. Validation

- `check_example`: hand-computed 2×6, `B=2` case checking `perm` and the permuted weights.
- `check_permutation_and_copy`: `perm` must be a permutation of `0..H-1`, and `permuted` must equal `W[:, perm]`.
- `check_zigzag_pattern`: the top three ranks fall in three different blocks, and the next three revisit those blocks in reverse order.
- `check_matches_oracle`: five shapes/block sizes (including `B=1` and `B=H`) against a pure-Python oracle.
- `check_ties_stable`: equal keys keep ascending original index, including the `B=1` and `B=H` degeneracies.
- `check_block_order`: each block is sorted by key descending.
- `check_validation`: illegal dimensions, shapes, types, non-finite values and `block_size` raise `ValueError`.
