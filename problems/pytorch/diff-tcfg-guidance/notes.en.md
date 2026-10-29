# In Depth: Tangential CFG with SVD Truncation

## Origin

*Tangentially Corrected Classifier-Free Guidance* (arXiv:2503.18137). The implementation follows HuggingFace diffusers' `src/diffusers/guiders/tangential_classifier_free_guidance.py`: `normalized_guidance` (L132–151) stacks the two predictions into `(B, 2, D)`, runs `torch.linalg.svd(..., full_matrices=False)`, zeroes the second row of `Vh`, and multiplies `uncond` by it — equivalent to keeping only the projection onto the first right singular direction. The extrapolation base is the projected `uncond`.

## Core points and pitfalls

1. **It is not "project the update onto the tangent plane"**: the usual one-line summary, but the code projects \(\epsilon_u\) **itself** onto the dominant direction and extrapolates from that point. The two readings give different results; the source wins.
2. **With two rows you never need a real SVD**: \(v_1\) must lie in \(\{\epsilon_c,\epsilon_u\}\). Write \(v_1=\alpha\epsilon_c+\beta\epsilon_u\); then \((\alpha,\beta)\) is the eigenvector of the \(2\times2\) Gram matrix \([[\|c\|^2,\,c\!\cdot\!u],[c\!\cdot\!u,\,\|u\|^2]]\) for its largest eigenvalue — that Gram matrix is exactly \(A^\top A\) expressed in the (non-orthonormal) basis \(\{\epsilon_c,\epsilon_u\}\). The closed form matched `torch.linalg.svd` bit-for-bit on 200 random inputs.
3. **Tied singular values leave the direction undefined**: when \(p=q\) and \(c\cdot u=0\) the two singular values coincide and the SVD result depends on the LAPACK path. This problem fixes the convention to \(\epsilon_c\)'s direction, giving a zero projection. In the closed form both candidate eigenvectors vanish at once, so the degenerate case needs explicit handling or you get 0/0.
4. **A scalar multiple is enough for the projection**: \(p=\frac{\epsilon_u\cdot w}{w\cdot w}w\), with \(w\) unnormalized. Normalizing first also works but adds a division that blows up when \(w=0\).
5. **The extrapolation base is the projected point**: \(s=1\) gives \(p+(\epsilon_c-p)=\epsilon_c\) bit-for-bit, the most reliable self-check.
6. **Parallel branches collapse to bare CFG**: \(v_1\) is the shared direction, the projection leaves \(\epsilon_u\) untouched, and the result equals plain CFG — which also shows TCFG only does something when the two branches disagree in direction.

## Verification approach

The reference solution uses the closed-form \(2\times2\) Gram eigendecomposition in float64 and never calls SVD; the judge's `_oracle` instead follows diffusers literally with `torch.linalg.svd` (float64). The two paths are independent, and the random comparison at `atol=1e-5` currently measures zero difference. Four more structural groups: the \(s=1\) identity, residual orthogonality at \(s=0\), the tied/parallel/zero-vector degeneracies, and the linearity of \(\text{out}-p\) in \(s\).

## Complexity

The closed form is \(O(D)\) per sample (inner products plus two scalar scalings); going through SVD is also \(O(D)\) for a 2-row matrix — shorter to write, higher constant.
