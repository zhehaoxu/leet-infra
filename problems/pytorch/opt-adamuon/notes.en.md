# In Depth: AdaMuon Element-wise Adaptive Orthogonal Update

## Source

- The algorithm: Si et al., *AdaMuon: Adaptive Muon Optimizer*, arXiv:**2507.11005** (verified), official implementation `Chongjie-Si/AdaMuon`. AdaMuon's two-stage design: element-wise second-moment adaptivity on Muon's orthogonal update \(O\) (\(\hat V\) EMA + RMS division), followed by **norm restoration** that keeps the whole-step displacement's Frobenius norm. The judge environment is offline; **the algorithm is defined by the problem statement and starter comments**. The deltas from the official implementation are explicit here: this problem **does not** apply Muon's \(\max(1,\text{rows}/\text{cols})^{0.5}\) spectral scaling (the restoration step cancels it), and 1-D parameters take a pure AdamW path.
- Library side: kozistr/pytorch_optimizer ships `Muon` (NS as a hand-written loop with quintic coefficients \((3.4445,-4.7750,2.0315)\), spectral scaling switchable); the in-library AdaMuon form is **to be re-sourced**. This problem pins NS to the pseudocode in the starter comments — transpose handling for rows>cols, and normalization by \(\|X\|_F+10^{-7}\) before iterating.
- One-line summary: **AdaMuon = Muon's "fairness across the spectrum" + Adam's "adaptivity across coordinates", with norm restoration gluing both onto the total scale \(\text{lr}\cdot\|O\|_F\)**. The per-coordinate division changes the distribution, the restoration changes the total — neither step works without the other.

## Full derivation and index convention

At step \(t\) (per-parameter counter starting from 1; the 2-D and 1-D paths each use their own counter for bias correction), the four transforms of the 2-D path each own one job:

- **Momentum + Nesterov**: \(m_t=\mu m_{t-1}+(1-\mu)g_t\), \(G=(1-\mu)g_t+\mu m_t\). Nesterov uses the **updated** \(m\) — extrapolate after the EMA; the reverse order is plain momentum.
- **NS orthogonalization**: the quintic iteration \(X\leftarrow aX+(bA+cA^2)X\) (\(A=XX^{\!\top}\)) pushes \(G\) toward an orthogonal matrix with all singular values ≈1. Three engineering details: transpose when rows>cols (the iteration is stable for "tall" matrices); normalize \(X\leftarrow X/(\|X\|_F+10^{-7})\) before iterating (the quintic is scale-sensitive); transpose back afterwards. **Output scale**: NS approximately preserves the Frobenius norm but **not exactly** — \(\|O\|_F\) is a number determined by the iteration itself, not 1; that is exactly why the restoration must explicitly multiply by \(\|O\|_F\).
- **RMS second moment (driven by \(O\)!)**: \(v_t=\beta_2v_{t-1}+(1-\beta_2)O^{\odot2}\), \(\hat V=v_t/(1-\beta_2^t)\). The Ada preconditioner sits **after** orthogonalization — it tunes "within the already-fair directions of the spectrum, which coordinates move more", not the gradient itself as AdamW's \(g^2\) does.
- **Preconditioning + restoration**: \(U'=O/(\sqrt{\hat V}+\varepsilon)\) is per-coordinate (coordinates with large \(\hat V\) are shrunk); how much \(\|U'\|_F\) shrinks relative to \(\|O\|_F\) is data-dependent; the restoration \(U=U'\cdot\|O\|_F/(\|U'\|_F+10^{-12})\) pulls the total back to \(\|O\|_F\). Combined:
  \[\|\Delta\theta\|_F=\text{lr}\cdot\|U\|_F=\text{lr}\cdot\|O\|_F\quad(\text{always holds without decay; asserted every step}).\]

**The first-step sign plate**: \(v_0=0\Rightarrow\hat V_1=(1-\beta_2)O^2/(1-\beta_2)=O^{\odot2}\Rightarrow\sqrt{\hat V_1}=|O|\Rightarrow U'=\mathrm{sign}(O)\) (element-wise \(\pm1\), exact for small enough \(\varepsilon\)) ⇒ \(U=\mathrm{sign}(O)\cdot\|O\|_F/\sqrt{n}\) (\(n\) = element count) ⇒ **first-step displacements are element-wise equal**. In the example that magnitude is 0.047864 = \(\text{lr}\cdot\|O\|_F/\sqrt6\). This simultaneously pins "v's driving signal is \(O\)" and "restoration comes after the RMS division" — either one wrong (say \(v\) eats \(g^2\), or the norm recomputed before restoration) and the first step stops being equal-magnitude.

**The 1-D path**: pure AdamW (\(\beta_1,\beta_2\) from `betas`), independent of `momentum` and `ns_steps`. Orthogonalization is meaningless on a vector (nothing to make "spectrally fair"); the official method and this problem both fall back to Adam. The judge compares bit-for-bit against native `torch.optim.AdamW`.

**Where the spectral scaling went**: original Muon multiplies \(\max(1,\text{rows}/\text{cols})^{0.5}\) after NS (sometimes with an extra empirical 0.2) to bring the update RMS to an Adam-comparable level. AdaMuon's restoration **already** pins the total scale, so the spectral factor would be fully cancelled by the restoration (multiplying it changes nothing in \(\Delta\theta\), but does perturb \(v\)'s statistics) — hence explicitly disabled here; stacking both is the classic double-scaling mistake.

## Key points and common mistakes

1. **\(v\)'s driving signal is \(O\), not \(g\).** Putting the Ada preconditioner back on \(g^2\) (= momentum swap on AdamW) is a different problem — both the first-step sign plate and the random-sequence comparison fail.
2. **Restoration must recompute \(\|U'\|_F\) after the division.** Dividing \(\|O\|_F\) by a precomputed \(\sqrt n\), restoring twice, or restoring before the RMS division all break \(\|\Delta\theta\|_F=\text{lr}\cdot\|O\|_F\).
3. **Nesterov uses the updated \(m\).** Computing \(G\) before updating \(m\) is plain momentum — identical on the first step, diverging from step 2, caught by the long-sequence comparison.
4. **NS's three-piece kit**: transpose handling, pre-iteration normalization, and the quintic coefficient order \((a,b,c)=(3.4445,-4.7750,2.0315)\). Missing normalization makes the quintic diverge or converge to the wrong spectrum; missing transposition destabilizes wide matrices. The judge's "independent NS recomputation of \(\|O\|\)" requires your NS to match the in-test implementation step for step.
5. **No spectral scaling** (see derivation) — stacking it is double-scaling.
6. **No orthogonalization for 1-D**: plain AdamW; \(\beta_1\) and \(\mu\) are two different parameters, do not conflate them.
7. **Decoupled decay first, no-grad fully skipped**: consistent with the collection convention; decay first, and the \(\|O\|\) invariant is asserted in the no-decay configuration.
8. **float32's three ε floors**: NS normalization \(10^{-7}\), restoration denominator \(10^{-12}\), RMS denominator the argument \(\varepsilon\) — do not cross-wire them. The sign-plate assertion sets \(\varepsilon=10^{-12}\), leaving the slack to float32.

## Differences from neighboring problems

- **vs `opt-muon`**: Muon is "momentum + NS + spectral scaling", treating all coordinates alike; AdaMuon swaps the spectral scaling for "RMS preconditioning + norm restoration". Decidable difference: Muon's \(\|\Delta\theta\|\) is fixed by the spectral coefficient and \(\|O\|\) (a data-independent constant scale), while AdaMuon's per-coordinate displacement distribution evolves with \(v\) at constant total — under the same gradient sequence the two problems' element-wise trajectories diverge from step 2 on.
- **vs `opt-adamw`**: the 1-D path degenerates bit-for-bit; on 2-D the difference is twofold — the driving signal (\(g^2\to O^2\)) and the total constraint (Adam has none, AdaMuon has \(\text{lr}\cdot\|O\|_F\)). AdamW's "big-gradient coordinates take small steps" becomes "big-\(\|O_i\|\) coordinates take small steps, at constant total per step".
- **vs `opt-sophia2`**: both consume second-order/structural information to reshape the step, from opposite directions — Sophia's \(H\) is **fed from outside** (curvature is data), AdaMuon's \(O\) is **computed inside** by NS (orthogonalization is a deterministic function); and Sophia bounds with a clip (element-wise hard cap), AdaMuon with norm restoration (a soft global constraint, no per-element cap).
- **vs `opt-adafactor`**: both touch the second moment, in opposite directions — adafactor compresses state (\(v\) factored into rank-1 row/column sums), AdaMuon expands the signal (computing an extra \(O\) and storing another \(v\)). One saves memory, the other trades for quality.

## Cross-check convention and complexity

- Main criterion: the independent recompute \(\_oracle\_adamuon\) — no Optimizer class: 2-D via a functional NS + vectorized tensor ops, 1-D via float64 scalar loops replicating AdamW; mixed-shape parameters (5×2, 3×4, 6) compared over a 12-step random sequence with `allclose(atol=1e-5, rtol=1e-5)`.
- Hard reference: 1-D parameters match **native `torch.optim.AdamW`** bit-for-bit (\(10^{-6}\)).
- Structural assertions: first-step sign plate (equal magnitudes, tolerances \(10^{-4}\)/\(10^{-3}\)); without decay \(\|\Delta\theta\|_F=\text{lr}\cdot\|O\|_F\) each step (\(O\) recomputed with the in-test independent `_ns`, relative slack \(10^{-4}\)); a `grad is None` parameter stays bit-identical.
- Complexity: 2-D steps cost \(O(\text{numel}+ns\_steps\cdot\text{matmul})\) (each NS iteration two matrix products of \(X\times X^{\!\top}\) shape); 1-D matches AdamW. State equals AdamW's (\(m,v\) each once); NS intermediates do not persist in memory.
