# Walkthrough: Muon Newton-Schulz Orthogonalized Momentum

## Source
Keller Jordan, *Muon: MomentUm Orthogonalized by Newton-schulz* (2024, blog kellerjordan.github.io/posts/muon; implementation KellerJordan/Muon, used for the Modded-NanoGPT training record). This problem aligns with its single-device `SingleDeviceMuon` + `zeropower_via_newtonschulz5`, with the original bfloat16 iteration switched to float32 to make CPU judging deterministic.

## Core idea
SGD-momentum produces a low-rank, anisotropic update matrix \(G\). Muon "orthogonalizes" it into the closest semi-orthogonal matrix \(UV^\top\) (where \(G=USV^\top\)), which amounts to pulling every singular value up to 1 — updating all directions with equal magnitude, which empirically speeds up training of matrix layers considerably. SVD is slow though, so a **Newton-Schulz iteration** is used to approximate the matrix's zeroth power \(G(G^\top G)^{-1/2}\).

## quintic Newton-Schulz
The coefficients \((a,b,c)=(3.4445,-4.7750,2.0315)\) maximize the slope of the iteration \(X\mapsto aX+(bX+cX^3)_{\text{quintic form}}\) around 0, so 5 steps alone are "good enough" (it does not converge to exactly 1; the singular values land in \(\sim[0.5,1.5]\), which is lossless in practice). The iteration is written as:
\[ A=XX^\top,\quad B=bA+cA^2,\quad X\leftarrow aX+BA. \]
To make \(XX^\top\) square with spectral norm \(\le1\):
- transpose first when rows > columns (so \(X\) has width \(\ge\) height and \(XX^\top\) has the size of the smaller dimension), and transpose back at the end;
- normalize by the **Frobenius norm** (\(\lVert X\rVert_F\ge\) the spectral norm, so dividing by it guarantees spectral norm \(\le1\) and a stable iteration), adding \(10^{-7}\) to guard against the zero matrix.

## Spectral scaling
After orthogonalization each singular value of \(O\) is \(\approx1\); a true semi-orthogonal \(R\times C\) matrix (\(R>C\)) has Frobenius norm \(\sqrt{C}\). Multiplying by \(\max(1,R/C)^{0.5}\) aligns the update magnitude with the Adam-style scale of "more rows should also mean a larger update" (KellerJordan's `adjust_lr` convention).

## Pitfall in the Nesterov combination
```
buf = μ·buf + (1−μ)·g
G   = (1−μ)·g + μ·buf     # uses the updated buf
```
i.e. the nesterov variant: \(G=(1-\mu)g+\mu\,buf_{new}\). Dropping the \(g\) half of the term, or using the old buf, gives a different result.

## Comparison setup
The external reference is an equivalent vectorized recomputation (same quintic iteration, same transpose branch, same Frobenius normalization). Non-square shapes (\(5\times2\), \(2\times3\)) cover both the transpose and the no-transpose branch. In float32 the same operator order matches element by element (the example values measure maxdiff ≈ 1e-7, comfortably inside the 1e-5 tolerance). Parameters with `grad is None` are skipped.

## Note
Muon should only be applied to the hidden 2D weight matrices; embeddings, the output head and scalars/biases are better left to AdamW — which is why this problem restricts its input to 2D matrices.
