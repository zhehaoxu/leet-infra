# Notes · RoAD's Grouped 2-D Rotations

## Sources

- `huggingface/peft`, `src/peft/tuners/road/layer.py` (RoAD adapter): `_prepare_cols` builds `first_col` / `second_col` per variant, `_get_delta_weight` assembles the rotation matrix as "diagonal + swapped-half negated", and `_apply_road` performs inference with the elementwise `x * first_col + rotate_half(x) * second_col`. This exercise follows that implementation line by line.
- The RoAD paper id was not re-sourced; the formulas follow the code, and these notes only document the convention, without claiming word-for-word agreement with the paper.

## Core points and pitfalls

**Pairs are first half ↔ second half within a group, not neighbors.** With `group_size=4`, position 0 pairs with 2 and 1 pairs with 3; that keeps a pair on one rank after tensor-parallel sharding. Interleaved pairing misplaces everything once `group_size` grows.

**`road_1` needs `repeat_interleave(2)`, not `repeat`.** `[θ0, θ1]` must become `[θ0, θ0, θ1, θ1]` (used once in each half); `repeat(2)` yields `[θ0, θ1, θ0, θ1]` and breaks the angle-to-pair mapping.

**The second column's signs are easy to flip.** The block view is `R[a, b] = -second_col[a]` and `R[b, a] = second_col[b]`; both signs use the `second_col` at their own index. Using `a` for one and `b` for the other is invisible in `road_1` (where the two entries coincide) but wrong in `road_2`.

**`road_2` with non-uniform `alpha` is no longer a rotation.** Only identical scales of 1 make the block a true orthogonal rotation `α[[c,−s],[s,c]]`; per-position scales change row/column lengths and break `RᵀR = I` — this is deliberately kept as a check.

**`road_4`'s two angles come from different segments.** After `reshape(-1, 2, group_size)`, `[:, 0, :]` feeds `cos` and `[:, 1, :]` feeds `sin`; reusing a single angle collapses it into a special case of `road_2`.

**The elementwise form must match the matrix.** The source uses the elementwise path to avoid materializing an `out×out` matrix; the problem validates both paths, so any mismatch surfaces on `y == x @ Rᵀ`.

**Leading dimensions.** The source assumes a 2-D input; this problem allows any leading dims, so flatten them first, compute, and reshape back; `rotate_half` only swaps along the last dim.

## Related problems

- `peft-oft-cayley`: OFT left-multiplies the weight by a block-diagonal orthogonal matrix; RoAD rotates the **output** in 2-D pairs with only angles and scales.
- `peft-hra-householder-reflection`: HRA right-multiplies the weight by a product of reflections; this problem applies per-pair 2×2 rotations.
- The RoPE-style rotation in diffusion problems such as `diff-rf-timestep-shift` only touches position encodings; this problem reuses the same "swap half, negate" idiom inside an adapter.
