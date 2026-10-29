# Notes: An Unidentifiable Degree of Freedom Has to Be Projected Out

## Origin

The weight_norm backward is the archetype of "differentiate through a reparameterization" and a lasting question from the StyleGAN/Seq2Seq era. The same geometry reappears in today's LLM engineering in other guises — **the LoRA \(BA\) factorization, spectral normalization, and any writing where one degree of freedom is shared by two parameters** all need the same projection to disambiguate. Interviewers use it to see whether you understand that "the gradient is not unique ⟺ the parameterization is redundant".

## Why dV Must Be Orthogonal to v

Handle one slice, \(w=g\,v/\lVert v\rVert\), by splitting a perturbation of \(v\) into a component along \(\hat v\) and an orthogonal one. The along-\(\hat v\) part only changes \(\lVert v\rVert\), which is exactly what changing \(g\) does, so it cannot be identified from \(dw\) alone. With \(\langle a,b\rangle=\sum_j a_jb_j\):
$$dg=\langle dW,\hat v\rangle,\qquad dV=\frac{g}{\lVert v\rVert}\big(dW-\hat v\,dg\big)$$
and therefore \(\langle dV,v\rangle=\frac{g}{\lVert v\rVert}(\langle dW,v\rangle-\lVert v\rVert\,dg)=0\) holds identically — the sharpest self-check in the judging suite.

## How This Differs from "Just Differentiate W"

If autograd walks the graph `W = g * V / V.norm()`, does the resulting \(dV\) keep a component along \(v\)? No: \(\partial\lVert v\rVert/\partial v=\hat v\) cancels precisely the collinear part, which is exactly the formula above. That is why the problem can be cross-checked against autograd directly, and where the phrase "project it out" comes from.

## Core Ideas and Pitfalls

- **Folding axis**: for `(out, in, kh, kw)` the norm must fold every axis except axis 0; a wrong `dim=` silently normalizes elementwise.
- **Zero-norm slice**: the direction is undefined when \(\lVert v\rVert=0\); PyTorch guards with an epsilon, whereas this problem must raise ValueError.
- **\(dg\) uses \(\hat v\), not \(v\)**: forgetting the division inflates \(dg\) by a factor of \(\lVert v\rVert\).
- **Separate dtypes**: `dg` follows `g` and `dV` follows `V`, which may differ under mixed precision.

## How It Differs from Neighboring Problems

`ag-layernorm-backward` also subtracts a projection (along \(\hat x\)) but that one is a normalization constraint; here the projection comes from parameter redundancy and would exist even without a normalization axis. `ag-conv2d-backward` differentiates a bare weight and involves no reparameterization.

## Cross-Check Basis and Complexity

The oracle builds `W = g.unsqueeze(1) * V / V.norm(dim=..., keepdim=True)` and takes `[dg, dV]`, atol=1e-9 in float64; two property cases — "rescaling a slice of \(V\) changes nothing" and "\(dV\perp v\)" — cross-validate the implementation. In the example both slices have \(\lVert v\rVert=5\) so all four numbers are mental arithmetic. Complexity is \(O(\text{numel})\) with two reductions per slice.
