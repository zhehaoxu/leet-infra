# Solution Notes · 4-bit Tile-Interleaved Layout

## Sources

- GPTQ (Frantar et al., arXiv:2210.17323) packs eight 4-bit codes per int32 along the input dimension, and packs `qzeros` with the same nibble rule — the origin of "codes and zero-points share one slot rule".
- Marlin (the 4-bit inference kernel from IST-DASLab, as used by vLLM / llm-compressor) adds a 16×16 tile grid and a fixed interleave of tile-local columns, so that a row's 16 codes come back in one 128-bit load. **Marlin has no paper**; it is a kernel implementation technique. This problem abstracts its layout into explicit rules for index bookkeeping and does not claim byte-level equality with any upstream revision (re-check that revision's `pack` implementation if exactness is needed).
- AWQ (Lin et al., arXiv:2306.00978) packs via `WQLinear` and reorders channels when fusing QKV — the real-world prototype of the `perm` layer: storage-column order is not logical-column order.

## Key points and pitfalls

**All four coordinate levels matter.** The int32 position is `(tile row, tile column, row-in-tile, half-word)`. Drop the tile row and the second tile reads tile 1's data once `K > 16` (`check_tile_boundary` targets exactly this).

**Slot order is `ORDER = [0,2,4,6,1,3,5,7]`.** Low/high nibbles of a byte carry local columns `(j, j+4)`, so read slot `ORDER[j]` per column (or invert to `INV[b] = [0,4,1,5,2,6,3,7]` per slot). Both are correct; mixing them permutes every column.

**`perm` scatters, it does not gather.** The meaning is "data of storage column `c` belongs to logical column `perm[c]`", i.e. `W[:, perm[c]] = …c…`. Writing `W[:, c] = base[:, perm[c]]` is the inverse permutation, which is invisible for involutions such as reversal — `check_perm_scatter` pins the direction with a four-way interleave `[0,4,8,12,1,5,9,13,…]`.

**Zero-points are storage-side, scales are logical-side.** `qzeros` is packed alongside the weights and lives in storage columns; `scales` are per logical channel. Mixing them shifts everything whenever `perm` is not the identity.

**Carry int32 shifts in int64.** An int32 bit pattern may be negative, and `>>` is arithmetic; convert (or mask with `0xFFFFFFFF`) before shifting, otherwise the high slot drags in the sign bit.

**Shape checks.** `K` and `N` must be multiples of 16, otherwise the tile grid is incomplete; raise `ValueError` by contract.

## Cross-checking

- Reference implementation: a `(h, slot)` double loop broadcasts whole tile columns into `(K, N//16)`, then scatters with `centered[:, perm] = …` and multiplies by the scale.
- Independent oracle: decodes row by row, column by column, reading slot `ORDER[j]` for local column `j` — the opposite direction — and assembles with nested lists, sharing no control flow with the reference.
- Tests also include a separate encoder `_encode` (logical codes → int32) for round trips; 30 random shapes (`K, N ∈ {16,32,48}`) plus 8 targeted cases (identity perm, column slots, row slots, zero packing, tile boundary, perm direction, bad shape, example).
- Storage accounting for one 4-bit weight matrix: `K·N/2` code bytes plus 8 zero-point bytes per 16-column group (scales excluded here).

## Complexity

- `O(K·N)` bit operations, linear in the number of weights; millisecond-scale for `K, N ≤ 48`.
- In a real kernel this step is fused with dequantization and the GEMM's `ldmatrix`; only the coordinates and bit order are kept here.

## Related problems

- `quant-uint4-asym-pack`: only the byte-level packing ("even index low nibble, odd index high nibble"); this problem adds the tile grid and column permutation on top.
- `group-wise-quantize`: the numerical per-group quantization rule, no byte layout.
- `gptq-column-quantize`: GPTQ's column-wise error compensation; weights stay floating-point, no packing.
- `awq-activation-scale`: grid search over the scale exponent; this problem is downstream — how the scaled 4-bit codes reach memory.
