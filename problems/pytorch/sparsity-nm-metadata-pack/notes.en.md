# Solution Notes: 2:4 Sparse Metadata Packing

## Origin
The 2:4 semi-structured storage format comes from NVIDIA Ampere's Sparse Tensor Core documentation and the cuSPARSELt / CUTLASS implementations: compressed values plus 4 bits of metadata per group of 4 elements, with metadata rows 32-bit aligned. In this collection `sparsity-nm-2-4` only decides "which two survive" (the mask level), while this problem decides "how the survivors are described to the hardware" (the layout level) — adjacent stages of one pipeline.

## Key points and pitfalls
1. **Descriptor formula**: \(\text{nib}=i_0+((i_1-1)\ll2)\). The low 2 bits hold the **absolute** index of the first non-zero (0–2, never 3) and the high 2 bits hold the **relative** offset of the second (\(i_1-1\in\{0,1,2\}\)). Storing \(i_1\) directly overflows bit 4 when \(i_1=3\) and corrupts the next nibble.
2. **Ascending order**: both `values` and the descriptor use \(i_0<i_1\); the first value taken must be the smaller index. A stable sort such as `argsort((g==0))` yields ascending positions with non-zeros first in one shot.
3. **Row-wise padding**: \(G\) need not be a multiple of 8. Padding goes at the **end of each row**, not the end of the tensor; for \(G=3\), \(P=8\) and the five nibbles after the third group must be zero.
4. **Byte order**: even groups occupy the low 4 bits. Swapping them (high nibble for group \(2j\)) silently shifts every decoded index.
5. **Only zero-ness matters**: `-0.0` counts as zero; the format never depends on the magnitude.

## Verification
The reference reshapes to `(R, G, 4)`, takes the two ascending positions with `argsort((g==0), stable=True)[:,:,:2]`, `gather`s the compressed values, computes the nibbles, pads, and packs pairs into uint8. The judge's oracle scans groups in pure Python, records `!= 0.0` positions and assembles bytes by hand, then compares `values` and `meta` bit-exactly. Extra assertions: decoding `values` + `meta` back to a dense matrix (with a decoder implemented independently in the test) must reproduce the original exactly; for \(G=3\) the nibbles after the third group are all zero; the byte-count alignment table (\(K=4,8,32\to4\) bytes, \(K=36,64\to8\) bytes); ascending in-group order; and the extreme combination \(i_0=2,i_1=3\) encoding to nib=10 (hence byte=10).

## Complexity
\(O(RK)\) (one classification pass per element); the output is \(RK/2\) values plus \(RK/8\) bytes of metadata.
