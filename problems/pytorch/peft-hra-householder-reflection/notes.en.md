# Notes · HRA's Householder Reflections and Compact WY

## Sources

- `huggingface/peft`, `src/peft/tuners/hra/layer.py` (HRA adapter landed in 2024): `_get_householder_vt` produces `(V, T)`, `get_delta_weight` assembles `H = I - (V T) Vᵀ`, and `_apply_delta_weight_to_input` computes `x Hᵀ` on the input side. This exercise follows that implementation line by line; the HRA paper id was not re-sourced, so the code is the reference.
- The compact WY representation itself comes from Schreiber & Van Loan, *A Storage-Efficient WY Representation for Products of Householder Transformations* (SIAM J. Sci. Stat. Comput., 1989), which compresses `r` reflectors into `I - V T Vᵀ`.

## Core points and pitfalls

**The product order is `P_0 @ P_1 @ … @ P_{r-1}`, not the reverse.** The compact form only holds for this order. Using the reverse flips a whole block of signs in the example while still passing an orthogonality check — `HᵀH = I` alone cannot catch it.

**Take the strictly-upper triangle in `T_inv`.** `triu(VᵀV, diagonal=1)` drops the diagonal; the diagonal is already 1 (columns are unit length), supplied by the `0.5 I` term. Using `triu(·)` with the diagonal gives `0.5 + 1 = 1.5` there and corrupts `T`.

**`T` comes from a triangular solve, not from transposing `T_inv`.** `solve_triangular(T_inv, I, upper=True)` yields `T = T_inv^{-1}`; an explicit `torch.inverse` differs at the 1e-7 level, and this exercise is cross-checked against the triangular-solve convention.

**The GS branch does not reuse the same `T` machinery.** After orthonormalization `VᵀV = I`, the cross terms vanish and `T` is exactly `2I`; still solving the triangular system adds numerical noise on top of the exact answer.

**Merge is `W0 @ H`, not `H @ W0`.** HRA acts on the input side (`x Hᵀ`), so the weight is right-multiplied. Left-multiplication errors loudly on non-square shapes and silently computes garbage on square ones.

**Reverse is not an inverse.** `H` is orthogonal, so `H⁻¹ = Hᵀ`; the source obtains it directly via `T.T`. Both are numerically equivalent, but the `T.T` convention is what keeps unmerge bit-consistent with merge.

**QR signs are not unique.** In the GS branch `V = Q` depends on `torch.linalg.qr`'s convention; the checks only assert `H = I - 2QQᵀ` and `H u = -u` (both invariant to column signs), never the elementwise sign of `V`.

## Related problems

- `peft-oft-cayley`: OFT builds a **block-diagonal** orthogonal matrix with the Cayley transform `(I+Q)(I−Q)⁻¹` and left-multiplies the weight; this problem takes a product of Householder reflections and acts on the input (right-multiplication).
- `peft-loha-hadamard`: LoHa is a low-rank factorization with a Hadamard product; there is no Hadamard product here.
- `peft-lokr-factorization` / `peft-lokr-kron-merge`: Kronecker factorization; this problem is an orthogonal reflection.
