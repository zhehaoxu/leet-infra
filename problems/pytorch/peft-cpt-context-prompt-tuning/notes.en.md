# Solution notes — CPT

## Official account

- HF PEFT `peft/tuners/cpt/model.py` (`CPTEmbedding`); paper arXiv 2410.17222.
- Judging domain: mask-family parsing, epsilon radii, L2 ball projection, embedding
  combine, decay weighted loss. Hooks / gather / vocab I/O are out of scope.

## Pitfalls

- **Epsilons are fp32 throughout**: the official `torch.Tensor([...])` /
  `torch.Tensor(mask)` default to fp32 — anchors must use fp32 representations
  (0.05 → 0.05000000074505806, 1e-10 → 1.000000013351432e-10) with tight tolerances.
- **Shift semantics are easy to get wrong**: `labels[..., 1:]` and `mask[..., 1:]`
  shift together; the valid-position count is the non-(-100) entries of t-1; group ids
  are looked up on the **shifted** mask (a missing-shift mutant is caught via shape).
- **Decay iterates in reverse**: unique (ascending) then flipped — the largest group id
  gets weight 1, later groups multiply the factor. Forward iteration swaps weights.
- **Projection shrinks only**: `clamp(norm, min=eps)` keeps scale <= 1; the
  `clamp(max=eps)` mutant grows small-norm rows onto the ball surface — caught by the
  norms assertion.
- **Three true equivalences (removed)**: on the non-negative integer domain
  `(m>0)&(m%4==k)` equals `m%4==k` (0%4=0 never lands in 1/2/3); a zero row under a
  `>=0` mask has scale exactly 1 (eps/eps=1); addition commutes in embed.
- **Root cause of the proj-p1 survivor**: without a zero row in the main delta,
  `new_w[mask] *= (k,1)` broadcasts identically to `new_w *= (n,1)` — the main case
  must include an exactly-zero row so k != n triggers a broadcast error.
- `set_default_dtype(float64)` in the test must precede the F import; CE returns 0
  (not exclusion) at ignore_index=-100 positions, unlike the official
  `loss[bool]` filter — manual cross-checks must slice valid positions.

## Mutant list (14)

mask: %4==1→2, view transpose; eps: format/input swap, normalization denominator 1024,
base MIN→0; proj: clamp min→max (grows), whole-tensor scaling (broadcast), `>0` mask
removed as a proven equivalence; embed: dropped delta, swapped order (equivalence
removed); loss: unshifted mask, broken ignore_index, reverse→forward iteration,
doubled factor, inverted type check, mean→sum. 14/14 CAUGHT.
