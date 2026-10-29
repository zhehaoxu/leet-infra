# Quantization: Per-Group Asymmetric uint4 Pack and Unpack

## Description

Per-group asymmetric 4-bit quantization (as in AutoGPTQ) quantizes weights into uint4 integers \(q\in[0,15]\) and reconstructs them with a shared scale \(s\) and zero-point \(z\):

\[W_{ij}=s_g\cdot(q_{ij}-z_g).\]

To save memory, two uint4 codes are packed into one uint8: **the low 4 bits hold the even-index code and the high 4 bits hold the odd-index code**:

\[\text{buf}_k=(q_{2k}\ \&\ 0x\text{F})\ |\ ((q_{2k+1}\ \&\ 0x\text{F})\ll 4).\]

Given the packed uint8 buffer `packed`, per-group scales `scales`, and zero-points `zeros`, unpack and reconstruct the original floating-point weights `W`.

## Task

Implement `solve(packed, scales, zeros, group_size, N)`:

- `packed`: uint8 with shape \((M, \lceil N/2\rceil)\).
- `scales`: float32 with shape \((M, \lceil N/\text{group_size}\rceil)\).
- `zeros`: int64 with the same shape as `scales`, values \(0\dots15\).
- `group_size`: positive integer dividing \(N\).
- `N`: original number of columns.
- Return `W`: float32 with shape \((M, N)\).

## Hints

- Unpack first: `(packed & 0xF)` gives even-position codes, `((packed >> 4) & 0xF)` gives odd-position codes; interleave them into a \((M, N)\) uint4 tensor.
- Use column index `j // group_size` to look up the corresponding scale and zero-point.
- If \(N\) is odd, the high nibble of the last uint8 is padding and should be ignored.
