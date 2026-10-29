# Deep Dive: WaveFT Sparse Spectrum Delta Weights

## Source

peft `tuners/waveft.py` (WaveFT). `WaveFTLayer.update_layer` does only three things: validate \(n_\text{frequency}\), draw positions from the seed, and initialize the spectrum parameter. The actual delta is produced by `get_delta_weight`, which has two branches:

- the **direct path** (`use_idwt=False`): scatter the spectrum straight into a dense \((d_\text{out},d_\text{in})\) matrix and multiply by `scaling`;
- the **IDWT path** (`use_idwt=True`, peft's default): first pad the matrix to `d + reduction` using the family's size reduction \((r_H,r_W)\), even-ify the padded sizes, shift the original coordinates by `(pad-d)//2` to center them, split into `cA/(cH,cV,cD)`, hand them to `waverec2d`, then center-crop back to \((d_\text{out},d_\text{in})\) and again multiply by `scaling`.

This problem implements the direct path. The IDWT path depends on `waverec2d` (ported from PyTorch-Wavelet-Toolbox), which is not in the judging environment, so the scope is explicitly cut there — this is also the "faster but slightly worse" configuration in peft.

## Coordinate semantics (the traps)

1. **Flatten, then restore**: `randperm(d_out*d_in)` yields row-major flat indices; placement uses `idx // d_in` and `idx % d_in`. Swapping `d_in` for `d_out` in `divmod` breaks on non-square weights.
2. **The generator must be seeded**: only `torch.randperm(n, generator=generator)` reproduces peft; omitting the generator consumes global RNG state, so the same seed gives different positions depending on call order.
3. **`scaling` applies to the delta**: `delta = dense * scaling`, not to the spectrum before scattering (numerically equivalent, but peft adds the return value of `get_delta_weight` to the base weight).
4. **A full budget is not a `reshape`**: with \(n_\text{frequency}=d_\text{out}d_\text{in}\), `randperm` is a **permutation**, so `delta.flatten()` is the spectrum shuffled — not `spectrum.reshape(d_out,d_in)`. `check_full_budget_covers_every_cell` pins this down by comparing sorted values.

## Forward and merge

The unmerged `WaveFTLinear.forward` is `result + F.linear(x, delta)`; `merge` is `weight.data += delta` (transposed first when `fan_in_fan_out`). They are equivalent, so the reference returns both `delta` and the merged-output `y`, and the tests check the sparse structure as well as the numerics.

## Verification approach

- `check_example`: seed 777, \(n=4\), `scaling=25`, pinning both `delta` and `y`.
- `check_index_selection`: three shapes/seeds, recovering positions from `(delta != 0)` and comparing to `randperm`, with per-position value checks.
- `check_forward_is_merged_linear`: 3-D batch, with and without `b0`, asserting the forward equals a plain linear layer with \(W_0+\Delta W\).
- `check_zero_spectrum_identity`: peft initializes the spectrum to zero, so the delta must be strictly zero (`torch.equal`).
- `check_scaling_is_linear`: doubling `scaling` doubles both the delta and the forward delta.
- `check_seed_changes_support`: different seeds give different supports, proving the seed is actually used.

## Complexity

Building `delta` costs \(O(d_\text{out}d_\text{in})\) (the dense matrix occupies that much memory anyway), while trainable parameters are only \(n_\text{frequency}\); the forward adds one sparse delta term over the merged linear layer.
