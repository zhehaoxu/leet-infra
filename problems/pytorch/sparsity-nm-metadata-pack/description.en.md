# Sparsification: 2:4 Sparse Metadata Packing for Tensor Cores

## Problem

Sparse Tensor Cores since NVIDIA Ampere consume **2:4** semi-structured weights only: exactly 2 non-zeros in every 4 consecutive elements. The hardware never reads the dense matrix; it reads two arrays:

- **compressed values**: the 2 surviving values of each group packed into a vector half the original length;
- **metadata**: which two of the four slots those values came from.

Metadata costs **4 bits per group**: for in-group indices \(i_0<i_1\in\{0,1,2,3\}\), the low 2 bits store \(i_0\) (0–2) and the high 2 bits store \(i_1-1\) (0–2) — the 4-bit code space maps exactly onto the six ways of choosing 2 of 4 slots (the other codes go unused). Nibbles are then paired into bytes (even groups in the low 4 bits), and each row's nibble count is padded to a multiple of **8** (4 bytes, one 32-bit word) so every row's metadata starts 32-bit aligned.

This problem is encoding only: given a weight matrix that already satisfies 2:4, produce `values` and `meta`; no GEMM is involved.

## Rules

Given `weight` (float32, \((R,K)\), \(K\%4=0\), exactly 2 non-zeros in every 4 consecutive elements, non-zero meaning `!= 0.0`):

1. split the last dimension into \(G=K/4\) groups of 4;
2. `values`: within each group take the two non-zeros in **ascending index** order, giving \((R,2G)=(R,K/2)\);
3. per group compute the 4-bit descriptor \(\text{nib}=i_0\,|\,((i_1-1)\ll2)\);
4. pad each row's nibbles with zeros to \(P=\lceil G/8\rceil\times8\);
5. pack pairs into bytes: `byte[j] = nib[2j] | (nib[2j+1] << 4)`, producing `meta` (uint8, \((R,P/2)\)).

Return `(values, meta)`. `torch.sparse` / cuSPARSELt are forbidden.

## Examples

**Example 1**: `K=8` (two groups) with non-zero positions \(\{1,2\}\) and \(\{0,3\}\)

```
weight = [[0, 3, -2, 0,  5, 0, 0, 7]]

group0 non-zeros at i0=1, i1=2 -> nib = 1 | ((2-1)<<2) = 1 | 4 = 5
group1 non-zeros at i0=0, i1=3 -> nib = 0 | ((3-1)<<2) = 0 | 8 = 8

values = [[3, -2, 5, 7]]
G=2 -> P=8 nibbles (6 padded zeros) -> 4 bytes
byte0 = 5 | (8 << 4) = 5 + 128 = 133
meta  = [[133, 0, 0, 0]]
```

**Example 2**: `K=12` (three groups; the third lands in byte1's low nibble, the rest padded)

```
weight = [[1, 0, 0, 2,  0, 3, 4, 0,  5, 6, 0, 0]]
nib    = [8, 5, 0]        # 0|2<<2, 1|1<<2, 0|0<<2
byte0  = 8 | (5 << 4) = 88
values = [[1, 2, 3, 4, 5, 6]]
meta   = [[88, 0, 0, 0]]
```

## Constraints

- float32 input, uint8 output; **exactly** 2 non-zeros per group.
- Padding nibbles must be 0, and each row is always \(P/2\) bytes (4 bytes for \(G=1\ldots8\), 8 bytes for \(G=9\ldots16\)).
- Low nibble first: group \(2j\) occupies the low 4 bits of `byte[j]`.
- `values` + `meta` must reconstruct the original matrix losslessly.
