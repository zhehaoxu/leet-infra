# In Depth: Adafactor Factored Second Moment

## Source
Shazeer & Stern, *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost*, arXiv:1804.04235 (2018). The algorithm skeleton follows `AdaFactor` from kozistr/pytorch_optimizer (the BigVision variant, the eager per-param path, `ams_bound=False`), and adopts its row-normalized outer-product reconstruction of \(\hat v\) (the same shape as `_approximate_second_moment` in the HuggingFace/FairSeq Adafactor).

## Three ideas that save memory and remove LR tuning
1. **Factored second moment**: a matrix \(v\in\mathbb R^{r\times c}\) is approximated by the outer product of the row statistics \(v_r\in\mathbb R^{r}\) and the column statistics \(v_c\in\mathbb R^{c}\), shrinking the state from \(rc\) to \(r+c\). The reconstruction is written as
   \[ \hat v_{ij}=\frac{v_{r,i}}{\sum_k v_{r,k}}\,v_{c,j} \]
   that is, the row vector is first normalized into a "per-row share" and then multiplied by the column statistics. This normalization brings the magnitude of \(\hat v\) back onto the scale of \(v_c\).
2. **RMS update clipping**: the overall RMS of the preconditioned direction \(u=g/\sqrt{\hat v}\) is divided by \(\text{clip\_threshold}\) and clamped to \(\ge1\) before being divided back — which caps the RMS of each step's update at the threshold. This is the source of the "learning-rate-free" behavior.
3. **Step scaling by parameter RMS** (`scale_parameter`): \(lr_{eff}=\max(\epsilon_2,\operatorname{RMS}(p))\cdot lr\), so that large parameter matrices take larger steps.

## Conventions chosen for CPU determinism in this problem (must match the statement)
- **`relative_step` is disabled**: the given `lr` is used directly, avoiding the \(1/\sqrt t\) dependence on the step count (stated in the problem).
- **Momentum is float32**, not the reference library's default bfloat16 — otherwise the landing point across backends is uncontrollable.
- In the second-moment decay \(\beta_{2,t}=\min(\beta_2^{cap},1-t^{-0.8})\) the default decay_rate is \(-0.8\); the EMA weight is \(b=1-\beta_{2,t}\), i.e. `v = (1-b)*v + b*U`, equivalent to the standard \(v\leftarrow\beta_{2,t}v+(1-\beta_{2,t})U\).
- The first momentum is EMA'd **after clipping and multiplying by \(lr_{eff}\)**, and the final displacement uses \(m\) itself (`p -= m`), with no further lr multiplication.
- `eps1=1e-30` is added to \(g^2\) (equivalent to the reference implementation adding eps1 to the squared gradient up front rather than clamping later).

## Pitfalls
- The denominator of the reconstructed \(\hat v\) is \(\sum_k v_{r,k}\) (a sum over the **row statistics**), not the columns.
- The RMS used by update clipping is computed on \(u=g/\sqrt{\hat v}\) (after preconditioning, before multiplying by lr).
- The order of the first momentum, weight decay, and the update follows steps 5→9 of the statement.

## Cross-check convention
The external reference uses an equivalent vectorized recomputation (explicit row/column EMAs, outer-product reconstruction, RMS clipping, momentum). Measured on the three-step example values, solution and reference give maxdiff 0 (float32, same operator order). Parameters with `grad is None` are skipped.
