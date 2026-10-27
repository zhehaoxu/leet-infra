# PEFT: OFT Orthogonal Finetuning via Cayley Transform — In Depth

## 1. Formula and provenance

OFT (Qiu et al., *Controlling Text-to-Image Diffusion by Orthogonal Finetuning*, arXiv:2306.03889; PEFT `tuners/oft/layer.py`). Unlike the LoRA family's "additive increment", OFT learns an **orthogonal rotation**: \(W'=R W_0\) with \(R^\top R=I\). It preserves every output neuron's weight norm (spectrum invariant, energy invariant) and only turns directions — particularly friendly to the semantic structure of diffusion models.

## 2. Cayley parametrization: why skew-symmetric → orthogonal

For any skew-symmetric \(Q^\top=-Q\), let \(R=(I+Q)(I-Q)^{-1}\). Verification:

\[R^\top R=(I-Q)^{-\top}(I+Q)^\top(I+Q)(I-Q)^{-1}=(I-Q)^{-1}(I-Q)(I+Q)(I-Q)^{-1}=I\]

The middle step uses \((I+Q)^\top=I-Q\) and the **commutativity** of skew-symmetric matrices with the identity matrix (\(QI=IQ\) trivially). Geometrically, this integrates a "rotation angular velocity \(\Omega\)" into a "rotation \(e^{\Omega t}\)": the Cayley transform is a rational (Padé-(1,1)) approximation of the matrix exponential \(e^{Q}\), matching to first order as \(Q\to0\).

**Zero init**: \(v=0\Rightarrow Q=0\Rightarrow R=I\); at the training start the adapter is an identity rotation. The LoRA family zeroes the factors, OFT zeroes the rotation — the same "start without perturbing the base" philosophy.

## 3. PEFT implementation conventions

`OFTRotationModule` / `_pytorch_skew_symmetric`:

- Each block has \(h(h-1)/2\) parameters (strict upper triangle, no diagonal — the diagonal must be 0 anyway), flattened in `torch.triu_indices(h, h, 1)` row-major order;
- \(Q\): first `Q[rows, cols] = v`, then `Q = Q - Q.T` (automatically filling the lower triangle with the negatives and cancelling the diagonal to 0);
- Cayley: PEFT defaults to `use_cayley_neumann=True`, approximating with the **Neumann series** \(I+2Q+2Q^2+2Q^3+\cdots\) (`num_neumann_terms=5`). Since \((I-Q)^{-1}=I+Q+Q^2+\cdots\), the closed form expands as \((I+Q)(I-Q)^{-1}=I+2Q+2Q^2+2Q^3+\cdots\) — the series and the closed form are truncation/limit of the same matrix; this problem uses the closed form, which agrees with PEFT's truncated version to high order when \(Q\) is small;
- PEFT also has `coft` (cos-theta projection under a sphere constraint) and `block_share` (all blocks share one \(R_b\)) switches; not covered here;
- Block diagonal: \(R=\operatorname{blockdiag}(R_1,\dots,R_n)\); `check_block_diagonal_structure` asserts all off-diagonal blocks are zero.

## 4. Judging red line and self-checks

- **Assert only the values of \(R\) and \(W'\)**; never assert eigenvectors or rotation-angle signs — the parametrization and eigendecomposition of an orthogonal matrix may differ in sign/order across implementations, and cross-checking those directly would surely fail (a collection-wide red line).
- Orthogonality self-checks: \(RR^\top=I\), \(R^\top R=I\); **Frobenius norm conservation** \(\|RW_0\|_F=\|W_0\|_F\) (multiplication by an orthogonal matrix on either side preserves the spectral norm family).
- Numeric paths: the reference solution uses `inv`, the oracle solves \((I-Q)R=I+Q\) with `solve`; both agree to \(10^{-12}\) in float64, ample for the float32 output tolerance \(10^{-5}\).
- BOFT (2311.06243) generalizes the block diagonal into a product of butterfly factors; PSOFT trains only a low-rank center of \(R-I\) — related methods are handled outside this problem.
