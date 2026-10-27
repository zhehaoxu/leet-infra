# PEFT: rsLoRA Scaling and LoRA+ Divergent Learning Rates — Algorithm Notes

## rsLoRA: why the scale is α/√r

Kalajdzievski (arXiv:2312.03732) writes the LoRA forward as \(h=W_0x+\frac{\alpha}{r}BAx\) and asks: how does the update's scale behave as the rank \(r\) grows? The conclusion: \(\alpha/r\) makes the update's effective norm decay like \(1/r\), faster than the \(1/\sqrt d\) scale inherent to standard weight initialization, so at high rank the LoRA branch becomes "increasingly useless" and training destabilizes. Switching to \(\alpha/\sqrt r\) restores the scaling law of standard parameterizations. Paper experiments: at high rank (\(r\ge 16\)) rsLoRA gives clearly lower loss and more stable training; at low rank the two scales are nearly indistinguishable. Note rsLoRA **changes neither parameters, nor gradients, nor the optimizer** — it is only a forward constant, fully merge-compatible: \(\Delta W=\frac{\alpha}{\sqrt r}BA\).

## LoRA+: B-side learning rate ×16

Hayou et al. (arXiv:2402.12354) analyze LoRA's training dynamics with finite-width theory: \(A\) projects the input into the low-rank space (down-projection) and \(B\) projects back to the output space (up-projection); their gradient scales are naturally asymmetric, and sharing one \(\eta\) degrades the feature learning rate in some layers so that \(B\) barely learns. The fix is \(\eta_B=\lambda\eta_A\) with a recommended \(\lambda=16\) (ablations show 16 is optimal on most tasks and makes the choice of \(\eta_A\) more robust). LoRA+ is **orthogonal** to the scale choice (\(\alpha/r\) or \(\alpha/\sqrt r\)) and can be stacked.

## Combined convention of this problem (judged points)

One SGD step: \(A'=A-\eta_A g_A\), \(B'=B-\lambda\eta_A g_B\); the outputs are the two scalings of the updated factors' product \(B'A'\):

\[\Delta_\text{rs}=\frac{\alpha}{\sqrt r}B'A',\qquad \Delta_\text{vanilla}=\frac{\alpha}{r}B'A'\]

- `check_ratio_sqrt_r`: the two outputs are two scalings of the same \(B'A'\); at any nonzero element the ratio is exactly \(\sqrt r\) (\(r=9\) → ratio 3).
- `check_zero_grad_no_change`: with \(g_A=g_B=0\) nothing updates, so both outputs equal the scaled **original** \(BA\) — this pins the semantics "output = product of the updated factors", not "output = this step's gradient contribution".
- `check_a_side_unaffected_by_lam`: with \(g_B=0\), switching \(\lambda\) (1 → 16) leaves every element unchanged — \(\lambda\) only enters \(B'\)'s update, the minimal assertion of LoRA+'s asymmetry.
- In `check_example` \(g_A,g_B\) each have a single nonzero element, yet \(\lambda\)'s effect spreads through the **product coupling** of \(B'\) and \(A'\) to all columns of \(\Delta\) — you cannot learn this by looking only at the changed column of \(B\).

## Division of labor within the collection

- `peft-lora-forward`/`peft-lora-merge`: static \(\alpha/r\) forward and merge.
- `peft-adalora-budget`: changes the rank itself (budget reallocation); this problem keeps the rank fixed and changes the scale (rsLoRA) and the learning rate (LoRA+).
- In real training all three can be enabled together: AdaLoRA manages the rank, rsLoRA the scale, LoRA+ the optimizer.

## Verification approach

Hand computation in `check_example` (\(r=4\), \(\eta_A=0.1\), \(\lambda=16\), \(\alpha=8\)): only \(A_{00}\) has gradient on the \(A\) side and only \(B_{01}\) on the \(B\) side → \(A'_{00}=0.9\), \(B'_{01}=-0.6\) → \(B'A'=[[3.9,0.4]]\) → \(\Delta_\text{rs}=[[15.6,1.6]]\), \(\Delta_\text{vanilla}=[[7.8,0.8]]\); `check_matches_oracle` compares against a pure-python scalar-by-scalar recomputation (float64 internally, float32 out, tolerance \(10^{-5}\)).
