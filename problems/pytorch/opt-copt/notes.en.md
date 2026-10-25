# In Depth: C-Optimizer Continuous Cosine Gate with Clipping

## Source

- The algorithm: the **C-Optimizer (COPT)** technical report from ByteDance Seed: a **continuous** function of \(\cos(g,m)\) as the update gate plus amplitude control (sign-preserving clipping). **The paper id is to be re-sourced**: the arXiv:2506.01327 recorded in the todo registration was fetched and verified to belong to a different paper (federated class-incremental learning) — do not cite it; the offline judge environment also cannot retrieve the original, so **the algorithm is defined by the problem statement and starter comments**: \(\text{gate}=\mathrm{clamp}((\cos+1)/2,0,1)\), gate before clip.
- Library side: kozistr/pytorch_optimizer ships `COPT` (exact file and attribute naming **to be re-sourced**). The library's gate mapping may differ slightly; this problem pins the mapping to the linear interpolation \((\cos+1)/2\), whose 0/1 special case lands exactly on cautious.
- One-line summary: **cautious is an on/off switch ("move only where signs agree"), COPT is a dial ("how much to move, proportional to agreement")**. The two coincide near \(\cos=1\), and part ways in the adversarial region \(\cos<0\) — the switch cuts the step to zero, the dial gives a smooth braking curve.

## Full derivation and index convention

At step \(t\) (per-parameter counter starting from 1), five quantities divide the labor:

- **\(m,v\) (two EMAs)**: isomorphic to AdamW; both the direction and the adaptive step size come from here.
- **\(\cos\) (alignment)**: \(\langle g,m\rangle_F/(\|g\|_F\|m\|_F)\), **one scalar per whole tensor** (not per coordinate! cautious's 0/1 test is per coordinate, COPT's continuous gate is tensor-wide — the second fork between the two problems). It uses the **just-updated** \(m\): EMA first, cos second. The zero-norm convention is \(\cos=1\): neither \(g=0\) (no update request) nor \(m=0\) (cold start with \(g=0\)) should be suppressed.
- **gate**: \(\mathrm{clamp}((\cos+1)/2,0,1)\). cos∈[−1,1] maps linearly onto [0,1]; the clamp is only a numeric backstop. **Identity chain**: \(\cos=1\Rightarrow\text{gate}=1\Rightarrow\) the gate is the identity ⇒ the algorithm degenerates to AdamW (clipping aside). That is the entire content of the judge's "aligned ≡ AdamW" assertion.
- **\(u\) (Adam direction)**: bias-corrected \(m/(1-\beta_1^t)\) over \(\sqrt{v/(1-\beta_2^t)}+\varepsilon\), operator order identical to opt-adamw (divide by the correction factors first, add ε last).
- **gate vs clip ordering**: \(u\leftarrow u\cdot\text{gate}\) **then** \(\mathrm{clip}(u,-\rho,\rho)\). The reverse order (clip then gate) differs numerically: on adversarial directions with \(|u|>\rho\) and gate<1, clip-then-gate gives \(\rho\cdot\text{gate}\), while gate-then-clip may retain larger values within \(\rho\) — the judge's oracle follows the stated order, and the swapped order betrays itself in the random-sequence comparison.

Bounds and invariants:

- **Hard bound**: \(|\Delta\theta|=\text{lr}\cdot|u\cdot\text{gate}\ \text{clipped}|\le\text{lr}\cdot\rho\) (the clip gives the bound, the gate only shrinks further). Asserted every step in the no-decay configuration.
- **Soft brake**: with \(\cos<1\), gate<1 multiplies the displacement down. Under extreme opposition (\(\cos\approx-1\)) gate≈0 and the displacement ≈0 — the judge pins this braking curve with "20 steps of \(+g\), then one step of \(-0.5g\)": that step's displacement must be \(<10^{-7}\).
- **First step hugs the bound**: \(m_1=(1-\beta_1)g_1\) is strictly parallel to \(g\) ⇒ cos=1 ⇒ gate=1; after bias correction \(u\approx\pm1\), and if \(\rho<1\) every element is clamped ⇒ the displacement is exactly \(\pm\text{lr}\cdot\rho\). In the example, \(\rho=0.5\), lr=0.1, and every coordinate moves exactly 0.05 on the first step.

## Key points and common mistakes

1. **cos uses the updated \(m\)**. Computing cos before the EMA is the most common ordering mistake — it betrays itself on step 1 (the stale \(m=0\) triggers the "zero norm ⇒ cos=1" backstop, which looks right, then diverges from step 2 on).
2. **cos is a scalar, not per-coordinate.** Writing \(\cos_i=g_im_i/(|g_i||m_i|)\) produces a different "continuous cautious"; the tensor-wide comparison fails. Per-coordinate masking belongs to cautious; the tensor-wide gate belongs to COPT.
3. **Zero norm ⇒ cos=1, not 0.** "No information means no suppression"; cos=0 would force gate=0.5 and needlessly halve cold-start steps.
4. **The gate multiplies the bias-corrected \(u\).** Multiplying \(m\) instead, or multiplying \(u\) but forgetting bias correction, yields different algorithms — the comparison is fully sensitive to operator order.
5. **Gate before clip** (see derivation). And the clip preserves signs: `clamp` to \([-\rho,\rho]\), never zeroing negatives.
6. **Decoupled decay first, no-grad parameters skipped entirely.** Decay never enters the \(m,v\) statistics; a parameter with `grad is None` does not even decay (the judge asserts it stays bit-identical).
7. **No survival-ratio rescaling.** Cautious implementations commonly compensate with `u * mask * (numel / mask.sum())`; COPT's gate carries **no** compensation — the gate already is the continuous amplitude, compensating would double-scale.
8. **Where ε and β sit**: \(\sqrt{v/(1-\beta_2^t)}+\varepsilon\) (ε added after bias correction), same as opt-adamw; do not put ε inside the square root.

## Differences from neighboring problems

- **vs `opt-cautious`**: two quantizations of the same idea. Cautious: **per-coordinate** 0/1 mask (\(g_iu_i>0\)) + rescale by survival ratio; COPT: **tensor-wide** continuous gate (\((\cos+1)/2\)) + no compensation + an extra \(\pm\rho\) clip. Decidable difference: both degenerate to AdamW under constant-direction gradients, but when a single coordinate flips, cautious cuts only that slot while COPT — whose cos is a scalar — brakes the whole tensor at once; the "one flipped coordinate" case behaves differently under the two answers.
- **vs `opt-sophia2`**: both clip to \(\pm\rho\), but Sophia's clip acts on the curvature-preconditioned \(m/\max\{H,\varepsilon\}\) with no direction gate (early steps hug the bound almost every step); COPT's clip acts on the Adam direction \(u\) behind a cosine gate (the bound is rarely attained). "Does the displacement equal lr·ρ routinely?" separates the two trajectories.
- **vs `opt-adamw`**: COPT is a strict extension of AdamW — with gate=1 and no clipping they are bit-identical; it brakes on adversarial directions and passes on aligned ones. The two endpoint assertions (aligned ≡ AdamW; adversarial-step displacement <1e-7) bracket the gate's working range exactly.
- **vs `opt-lookahead`**: Lookahead does trust correction at the multi-step scale (slow-weight shrinkage), COPT does directional trust correction within a single step (cos gate); one changes "how steps fold back on each other", the other "what one step is made of". Composable.

## Cross-check convention and complexity

- Main criterion: the independent recompute \(\_oracle\_copt\) — float64 **scalar loops** element by element (no Optimizer class, no vectorization), compared against the reference solution (float32 vectorized ops) over a 12-step random sequence with `allclose(atol=1e-5, rtol=1e-5)`. The two paths share only "the stated formula".
- Hard reference: constant-direction gradients + huge \(\rho\) (\(10^9\), clip never binds) match **native `torch.optim.AdamW`** at \(10^{-6}\) — this simultaneously pins "cos=1 ⇒ gate is the identity" and "the AdamW part of the reference is not miswritten".
- Structural assertions: adversarial step (20 × +g then −0.5g) displacement \(<10^{-7}\); with no decay \(|\Delta\theta|\le\text{lr}\cdot\rho\) each step; a `grad is None` parameter stays bit-identical.
- Complexity: \(O(\text{numel})\) per parameter per step; over AdamW it adds three tensor-wide reductions (inner product, two norms) and one element-wise gate multiply — a constant-factor increase, memory identical to AdamW.
