# Solution Notes · DeLoRA's Row-Norm-Normalized Update

## Source

- `huggingface/peft`, `src/peft/tuners/delora/layer.py`: `_compute_delta` builds `delta = B @ diag(lam/r / (||A_i|| * ||B^j||)) @ A` and then scales by `w_norm` per input channel; `forward` uses the equivalent element-wise path `(x * w_norm) @ Aᵀ → * scaling → @ Bᵀ`; `reset_delora_parameters` freezes `w_norm` before training with `torch.norm(w.data, dim=0).detach()`.
- Paper ids were not re-sourced one by one; the formulas follow the source code (these notes explain conventions only and do not claim word-for-word agreement with the paper).

## Core Points and Pitfalls

**Rank comes from `A`'s row count, not `B`'s.** In the source, `A` is `(r, in_features)` and `B` is `(out_features, r)`, matching the "down then up" LoRA layout; reading it the other way puts `dim=1` / `dim=0` on the wrong axis and changes the shape of `scaling`.

**The floor is clamped on the norm, not on the squared norm.** `clamp(norm, min=1e-4)` guards all-zero rows/columns; squaring first or using a different floor changes the zero-row scale by orders of magnitude. One all-zero row in the tests pins this boundary down.

**The scale is data-independent but not constant.** `scaling_i` depends only on the current row norms of `A` and column norms of `B`, never on activations; but `A` and `B` keep changing during training, so `forward` recomputes it every call. Do not cache it at init.

**`w_norm` multiplies along columns (input channels).** It comes from the base weight's `dim=0` norms and has length `in_features`; applying it with `unsqueeze(1)` (row direction) silently shifts everything.

**`‖delta‖_F = lam / sqrt(r)` is conditional.** It holds exactly only when the rows of `A` and columns of `B` are unit vectors and mutually orthogonal, so the rank-1 components are orthogonal; in general only `‖delta‖_F <= lam` is guaranteed. The tests check both the orthogonal equality and the general bound.

**The merge path must agree with the element-wise `forward` path.** `x @ deltaᵀ` expands to `((x * w_norm) @ Aᵀ * scaling) @ Bᵀ`; moving `w_norm` to the other side of `A` or `B` breaks the equivalence.

**Precision.** Products of norms followed by reciprocals are sensitive in float32; compute internally in float64 and cast back at the output to keep random fuzz cases within tolerance.

## Division of Labor

- `peft-dora`: DoRA splits a weight into "column norm × direction" and trains only the direction; DeLoRA bounds the norm of a low-rank *update* without decomposing the base.
- `peft-lokr`, `peft-oft-cayley`: the constraints live in the parameterization (orthogonal/semi-orthogonal factors); here the constraint is on the norm of the output update, and the factors need not be orthogonal.
- `peft-hra-householder-reflection`, `peft-road-block-rotation`: sibling adapters that transform the weight via reflection products and grouped rotations; DeLoRA only does an additive low-rank update with scaling.
- Per-channel scales in the quantization problems are part of the number format; the `scaling` here is a capacity constraint, so the two derivations should not be mixed.
