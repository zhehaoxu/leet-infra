# Optimizer: Muon Newton-Schulz Orthogonalized Momentum

## Problem

Muon (MomentUm Orthogonalized by Newton-schulz, Keller Jordan, 2024; used in the Modded-NanoGPT record runs) is meant for the **2D weight matrices** inside a neural network. It first runs plain SGD-momentum and then **orthogonalizes** the momentum matrix into "the closest semi-orthogonal matrix" as the update direction — replacing \(U S V^\top\) with \(UV^\top\) from \(USV^\top\). To compute the orthogonalization efficiently it approximates the zeroth power with a 5-step **Newton-Schulz iteration**, avoiding an explicit SVD.

This problem asks you to **subclass `torch.optim.Optimizer`** and implement Muon, with `step()` updating parameters in place; the judge feeds a fixed sequence of gradients and checks whether the final parameters match an independent reference implementation element by element. To keep judging stable on CPU, Newton-Schulz runs entirely in **float32** (the original implementation uses bfloat16).

## Rules

Implement the factory function `solve(params, lr=0.02, momentum=0.95, ns_steps=5, weight_decay=0.0)`, returning an instance of a `torch.optim.Optimizer` subclass; `params` is a list of 2D weight matrices and `step()` updates them in place. For each parameter \(p\) with gradient \(g\) (step \(t\), \(t\) counted from 1):

1. Momentum buffer: \(buf \leftarrow \mu\, buf + (1-\mu)\, g\)
2. Nesterov combination: \(G \leftarrow (1-\mu)\, g + \mu\, buf\)
3. **Newton-Schulz orthogonalization** \(\operatorname{NS}(G)\) (see below), giving the semi-orthogonal direction \(O\)
4. Spectral scaling: \(scale = \max\!\big(1,\ \tfrac{\text{rows}}{\text{cols}}\big)^{0.5}\), \(O \leftarrow O\cdot scale\)
5. Decoupled weight decay (`weight_decay`≠0): \(p \leftarrow p\,(1 - lr\cdot\text{wd})\)
6. Parameter update: \(p \leftarrow p - lr\cdot O\)

**Newton-Schulz iteration \(\operatorname{NS}(G)\)**, quintic coefficients \((a,b,c)=(3.4445,\,-4.7750,\,2.0315)\):
- If \(\text{rows}>\text{cols}\): transpose first (write \(X\leftarrow G^\top\)), otherwise \(X\leftarrow G\);
- Normalize: \(X \leftarrow X/(\lVert X\rVert_F + 10^{-7})\) (\(\lVert\cdot\rVert_F\) is the Frobenius norm of the whole matrix, which guarantees spectral norm \(\le 1\));
- Iterate \(ns\_steps\) times: \(A = X X^\top,\; B = bA + cA^2,\; X = aX + BA\);
- If you transposed earlier, transpose back at the end.

Initial \(buf=0\); skip parameters with `grad is None`; calling an off-the-shelf Muon or doing an explicit SVD is **forbidden** — write the iteration by hand as above.

## Example

A single weight \(p=\begin{bmatrix}0.5 & -0.2 & 0.1\\ 0.3 & 0.4 & -0.1\end{bmatrix}\), `lr=0.1`, `momentum=0.95`, `ns_steps=5`, `weight_decay=0`, with three gradient steps:

```
g₁ = [[ 0.2,-0.1, 0.05],[ 0.1, 0.3,-0.2]]
g₂ = [[-0.15, 0.2, 0.1],[ 0.05,-0.1, 0.3]]
g₃ = [[ 0.1, 0.05,-0.2],[-0.3, 0.15, 0.1]]
p after three steps ≈ [[0.432205, -0.322988, 0.124788], [0.256377, 0.285218, -0.137443]]
```

## Constraints

- Input is float32, 2D matrices; non-square shapes (such as \(2\times3\), \(5\times2\)) must exercise the transpose branch correctly
- Newton-Schulz uses float32, not bfloat16; the Frobenius normalization adds \(10^{-7}\) to guard against zero
- Must match the independent reference step by step exactly (see notes.md for the setup)
