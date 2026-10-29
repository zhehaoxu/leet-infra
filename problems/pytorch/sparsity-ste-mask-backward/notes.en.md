# Solution Notes: STE and Soft-Threshold Reparameterization Backward

## Origin
STE applied to discrete stochastic units was analysed by Bengio, Léonard & Courville (*Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation*, 2013, arXiv 1308.3432). The soft-threshold reparameterization is WSN (Kusupati et al., *Soft Threshold Weight Reparameterization for Learnable Sparsity*, ICML 2021, arXiv 2002.10509): \(w\cdot\mathrm{clamp}(|w|/\tau,0,1)\) with \(\tau\) learnable, converging to a target sparsity during training.

## Key points and pitfalls
1. **STE multiplies by the mask, not by 1**: the name invites the wrong shortcut. Pruned positions must get exactly zero gradient, otherwise the weights are pulled back up and the pruning evaporates; only kept positions go straight through.
2. **The equality in `hard_ste`**: \(|w|=\tau\) belongs to the kept side; it is the boundary that keeps training and inference consistent. Using `>` makes the threshold flicker.
3. **WSN has an extra gradient term**: \(y=w\cdot m(w)\) is a product of two factors that both depend on \(w\), so the chain rule gives \(m+w\,\mathrm{d}m/\mathrm{d}w\). For \(|w|<\tau\), \(\mathrm{d}m/\mathrm{d}|w|=1/\tau\) and \(\mathrm{d}|w|/\mathrm{d}w=\mathrm{sign}(w)\), together giving exactly \(2|w|/\tau\). Dropping the second term yields "STE with a soft mask" and halves the gradient.
4. **Sign of \(\tau\)**: for \(\tau\le0\) the mask is identically 1 with no sparsity meaning, hence the \(\tau>0\) constraint.
5. **Upstream gradient only, no autograd**: this is exactly the shape of a hand-written backward (the `backward` of a custom `autograd.Function`), so derive the scalar formulas before coding.

## Verification
The reference branches on `mode`, computes the mask and gradient, and returns `(w*m, grad, m)`. The judge's oracle recomputes all three outputs element by element in pure Python. Additional assertions: in `hard_ste` pruned positions have exactly zero gradient, kept positions pass the upstream value through, and the equality boundary is kept; the `wsn` gradient is cross-checked against `torch.autograd` in float64 differentiating the *same formula* (the only autograd in the test, used as an independent check on the judge side); the `wsn` mask saturates to 1 for \(|w|\ge\tau\); and both modes satisfy \(y=w\odot m\).

## Complexity
\(O(N)\) element-wise; \(O(N)\) memory.
