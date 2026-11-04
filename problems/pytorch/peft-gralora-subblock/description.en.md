# PEFT: GraLoRA Granular Sub-block Forward and Block Merge

## Problem

**GraLoRA** (Granular LoRA, arXiv:2505.20355, NeurIPS 2025 Spotlight; official repo `SqueezeBits/GraLoRA`, now merged into HuggingFace PEFT as `tuners/gralora`) fixes a structural flaw of LoRA: raising the rank \(r\) widens the bottleneck to \(r\) as well, which tends to overfit. GraLoRA splits the weight matrix into \(k\) segments along both the output and input axes and gives each segment an independent pair of low-rank factors — the **bottleneck is cut into \(k\) pieces while the total parameter count stays exactly the same as a same-rank plain LoRA** (\(r\cdot(d_\text{out}+d_\text{in})\)).

This problem replicates the two deterministic accounts of the official `layer.py`:

- **Layout account** (`update_layer`): the usable rank splits into \(r_\text{gra}=r-r_\text{hyb}\) (optionally reserving a full-rank plain-LoRA bypass), each segment carrying \(\text{sub}=r_\text{gra}/k\); A is stored per **input segment** as \((k, d_\text{in}/k, r_\text{gra})\), B per **output segment** as \((k, r_\text{gra}, d_\text{out}/k)\); scaling \(=\alpha/r\) uses the **total rank** (hybrid included), the same convention as plain LoRA.
- **Forward account** (the double-einsum chain in `Linear.forward`): input segment \(n\) goes through \(A_n\) up to \(r_\text{gra}\) dims, the rank axis is cut into \(k\) segments and rearranged (a permute swapping the "segment" and "input-block" axes followed by a compound reshape), then contracted with the matching **row segment** of B's \(j\)-th output segment.

The equivalent block formula is the key to the structure: block \((j,n)\) of \(\Delta W\) is

\[W^{(j,n)} = B_j[n\cdot s{:}(n{+}1)\cdot s,:]^{\!\top}\, A_n[:,\,j\cdot s{:}(j{+}1)\cdot s],\qquad s=r_\text{gra}/k\]

Each block has rank only \(s\) — B's \(n\)-th row segment paired with A's \(j\)-th column segment. That is the "granular bottleneck".

## Rules

- `mode="split_dims"`: returns `{"gralora_rank", "sub_rank", "sub_in", "sub_out", "scaling"}`; \(r\le 0\) raises `ValueError`; \(r_\text{gra}\%k\ne0\) raises `ValueError` (official assert).
- `mode="param_layout"`: returns the shapes of A/B (plus the hybrid bypass \(A_g:(r_\text{hyb},d_\text{in})\), \(B_g:(d_\text{out},r_\text{hyb})\) when present) and the zero-initialized set — **B and \(B_g\) are all zeros** (so \(\Delta W=0\) at the start of training), A and \(A_g\) are kaiming (nonzero).
- `mode="forward_delta"`: a literal transcription of the einsum chain (dropout=0); input \(x:(B,L,d_\text{in})\), returns \(scaling\cdot\Delta y:(B,L,k\cdot(d_\text{out}/k))\).
- `mode="block_delta"`: assembles the dense \(\Delta W:(d_\text{out},d_\text{in})\) explicitly with the block formula.
- `mode="full_forward"`: the gralora branch \(+\;scaling\cdot x\,A_g^\top B_g^\top\); \(r_\text{hyb}>0\) requires both bypass factors, \(r_\text{hyb}=0\) rejects them.
- Everything in float64.

## Example

\(d_\text{in}=6, d_\text{out}=8, r=4, k=2, \alpha=8, r_\text{hyb}=0\):

```
split_dims -> {gralora_rank: 4, sub_rank: 2, sub_in: 3, sub_out: 4, scaling: 2.0}
param_layout -> A:(2,3,4), B:(2,4,4), zero_init=["gralora_B"]
```

For given \(x:(2,3,6)\), \(A\), \(B\) (see the test anchors), `forward_delta` returns shape \((2,3,8)\), element-wise equal to \(x\cdot\Delta W^\top\cdot scaling\) with \(\Delta W\) assembled by `block_delta`.

## Constraints

- Sizes must satisfy \(k \mid d_\text{in}\), \(k \mid d_\text{out}\), \(k \mid r_\text{gra}\) (implied by the sub-block sizes and the official assert).
- The two hybrid factors must be given together or not at all, consistent with \(r_\text{hyb}\); violations raise `ValueError`.
- `block_delta` must be mathematically equivalent to `forward_delta` (cross-checked in the tests).
