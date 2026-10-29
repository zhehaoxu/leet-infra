# Quantization: Decoding a 4-bit Tile-Interleaved Weight Layout

## Problem

4-bit weight kernels (the GPTQ / Marlin lineage) do not store codes in row-major order. So that one warp reads contiguous memory and every row of a 16×16 tile lands in the same address range, weights are cut into tiles, the columns inside a tile are **interleaved** in a fixed order, and eight 4-bit codes are squeezed into one int32. Operations that fuse QKV also apply a global output-channel permutation `perm`. So "unpacking" is not `byte & 0xF`: you must get four levels of coordinates (tile row, tile column, row-in-tile, half-word) and the nibble slot order right, scatter back to logical columns through `perm`, and only then dequantize with the logical-column scale and zero-point. Get one level wrong and you read another column's codes — numerically it looks like "just a bit less accurate", but the whole table is permuted.

Given a `qweight` packed in this layout, a `qzeros` in the same layout, `scales` in logical-column order and the column permutation `perm`, reconstruct the floating-point weights `W`.

## Rules

Let the original weight be `(K, N)`, with both `K` and `N` positive multiples of 16. The layout has four levels:

1. **Tile grid**: `K/16` tiles along rows and `N/16` tiles along columns. Tile row `t` covers original rows `16t … 16t+15`; tile column `g` covers **storage** columns `16g … 16g+15`.
2. **Half-words**: the 16 codes of row `r` inside a tile are split into two half-words `h = 0, 1`; half-word `h` holds tile-local columns `8h … 8h+7`.
3. **Slot order**: each half-word has 8 nibble slots from the low 4 bits upward; the code of tile-local column `j` sits in slot `ORDER[j]`, with `ORDER = [0, 2, 4, 6, 1, 3, 5, 7]`. Hence bytes `0..3` of an int32 carry local columns `(0,4)`, `(1,5)`, `(2,6)`, `(3,7)` as low/high nibbles.
4. **Zero-points follow the same rule**: `qzeros` has no row dimension; in the int32 of group `g`, half-word `h`, slot `b` holds the zero-point of storage column `c = 16g + 8h + ORDER[b]`.
5. **Column permutation**: `perm[c]` is the logical column of storage column `c`; `perm` is a permutation of `0 … N−1`.
6. **Dequantization**: `W[k, perm[c]] = (code(k, c) − zero(c)) × scales[perm[c]]`. Zero-points are indexed by **storage** column, scales by **logical** column.

## Task

Implement `solve(qweight, qzeros, scales, perm, K, N)`:

- `qweight`: int32, shape `(K//16, N//16, 16, 2)`.
- `qzeros`: int32, shape `(N//16, 2)`.
- `scales`: float32, shape `(N,)`, logical-column order.
- `perm`: `list[int]` of length `N`.
- `K`, `N`: original row and column counts.
- Return `W`: float32, shape `(K, N)`, logical-column order. Compute in float64 and cast to float32; tolerance `1e-5`.
- Raise `ValueError` when `K % 16 != 0` or `N % 16 != 0`.

## Example

`K = N = 16` (a single tile), logical codes `code[k, n] = (3k + n) % 16`, `zero[n] = n % 2`, `scales = [1.0]×8 + [0.5]×8`, `perm = [15, 14, …, 0]` (storage column `c` becomes logical column `15−c`).

Row 0, half-word 0 of the stored codes is `0,1,2,…,7`; after slot reordering it packs as:

```
slot (low→high):  j0  j4  j1  j5  j2  j6  j3  j7
code           :   0   4   1   5   2   6   3   7
int32          : 0x73625140          # bytes = 0x40, 0x51, 0x62, 0x73
```

The zero-points follow the same rule (`zero = 0,1,0,1,…`):

```
qzeros[0, 0] = 0x11001100
```

After dequantization (first 4 rows × 8 columns):

```
W[0:4, 0:8] =
  [[14.0, 14.0, 12.0, 12.0, 10.0, 10.0,  8.0,  8.0],
   [ 1.0,  1.0, -1.0, 15.0, 13.0, 13.0, 11.0, 11.0],
   [ 4.0,  4.0,  2.0,  2.0,  0.0,  0.0, 14.0, 14.0],
   [ 7.0,  7.0,  5.0,  5.0,  3.0,  3.0,  1.0,  1.0]]
```

(Logical column `n` reads the code of storage column `15−n`; the last 8 columns use scale 0.5, hence the halved values.)

## Constraints

- `qweight` and `qzeros` are int32 bit patterns: carry intermediate values in int64, otherwise a right shift of the high nibble drags in the sign bit.
- Slots follow `ORDER`, not natural column order; the two half-words cover tile-local columns `0…7` and `8…15`.
- `perm` scatters "storage column → logical column"; do not implement it as a reversed gather.
- Only the returned tensor's values and shape are judged; any implementation style is fine.
