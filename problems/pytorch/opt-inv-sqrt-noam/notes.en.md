# In Depth: Noam Inverse Square Root Learning Rate

## Source
- Vaswani et al., *Attention Is All You Need*, arXiv:1706.03762, **equation (5)**:
  `lrate = d_model**(-0.5) * min(step_num**(-0.5), step_num * warmup_steps**(-1.5))`, annotated with
  "step_num starts at 1". With the paper's \(d_{model}=512,\ warmup=4000\) the peak is about 0.0007.
  This problem's formula is verbatim identical; the only adjustment is that **steps are 0-based** (following the
  collection's "construction is already step 0" convention): for \(t\ge1\) it coincides with the paper exactly, and
  \(t=0\) is given by the linear branch as 0 (the point just before the paper's step 1).
- Second place reusing the same expression: Shazeer & Stern, *AdaFactor*, arXiv:1804.04235 section 4.1, whose
  relative step size is \(\min(s^{-1/2}, s/warmup^{3/2})\) (`opt-adafactor` embeds exactly this, but there it is an
  update magnitude rather than a standalone scheduler).
- External numeric reference: `transformers/optimization.py::_get_inverse_sqrt_schedule_lr_lambda`
  (adapted from big_vision): `timescale` defaults to `num_warmup_steps or 10_000`, and it returns \(t/w\) for
  \(t<w\) and \(\sqrt{w/t}\) for \(t\ge w\). Multiply that by the peak \(P=d^{-1/2}w^{-1/2}\) and you get Noam:
  \(P\cdot t/w=d^{-1/2}t\,w^{-3/2}\) and \(P\cdot\sqrt{w/t}=d^{-1/2}t^{-1/2}\) - both branches match term by term.
  Measured relative discrepancy is \(\sim2\times10^{-16}\) (pure float rounding), so it is a valid step-by-step
  reference.
- The expression in todos.md #90(5), \(d^{-0.5}\min(t^{-0.5},t\,w^{-1.5})\), is correct and needs no correction; the
  real gap was that no standalone Noam-type scheduler existed in the collection, which this problem fills.

## Core points and pitfalls
1. **You cannot plug \(t=0\) into the min form**: `0**-0.5` raises `ZeroDivisionError` in Python. The piecewise form
   is mandatory, with the linear branch giving \(lr(0)=0\); this is the most common way to fail here.
2. **The warmup exponent is \(-1.5\), not \(-0.5\)**: \(t\cdot w^{-3/2}\) equals \(w^{-1/2}\) at \(t=w\), which is
   exactly where the decay branch starts. Using \(w^{-0.5}\) inflates the warmup by a factor of \(w\) and creates a
   jump at the peak; the grader pins this with "the increment at \(t=w\) still equals the warmup common difference".
3. **Peak location**: the branches cross at \(t=w\), so `argmax(lr)==warmup_steps`, and the rising part is an
   **arithmetic** (linear) sequence, not a square-root ramp - reusing \(t^{-1/2}\) as the warmup is a classic error.
4. **Absolute learning rate, not a multiplier of base_lr**: unlike the ratio-style WSD/OneCycle schedules, Noam's
   \(lr(t)\) is fixed by \(d,w\) and independent of the optimizer's original lr (all param groups get the same value).
5. **No total steps, no clamping**: as \(t\to\infty\) it simply keeps decaying as \(t^{-1/2}\), always positive; do
   not add a `min_lr` and do not pin it after some horizon. The long-run invariant
   `lr(t)*sqrt(t) == d**-0.5` exists precisely to catch that improvisation.
6. **\(w=1\) is a legal degeneration**: the peak is hit at step 1 and everything after is pure decay; both
   `model_dim` and `warmup_steps` must be positive integers.

## Cross-check convention
- Independent recomputation (primary reference): `_oracle` evaluates the paper's min closed form
  \(d^{-1/2}\min(t^{-1/2},t\,w^{-3/2})\) directly (with \(t=0\) special-cased to 0), structurally different from the
  piecewise solution; tolerance \(10^{-12}\) (relative scale for tiny values), over six \((d,w)\) pairs including a
  4000-step warmup and \(w=1\).
- External reference: `transformers.get_inverse_sqrt_schedule` with `SGD(lr=d**-0.5*w**-0.5)` and
  `timescale=warmup_steps`, read for 40 steps under the same convention, `torch.allclose(rtol=1e-12, atol=1e-15)`,
  four configurations.
- The example numbers were produced by actually running the reference solution (`model_dim=64, warmup_steps=5`,
  written into the statement with 8 decimals).
- Degenerate cases / invariants: \(lr(0)=0\) without crashing; arithmetic warmup with difference
  \(d^{-1/2}w^{-3/2}\); the peak exactly at \(t=w\) with the increment continuous across it; the decay branch strictly
  decreasing and positive; at \(t=604\) still `lr*sqrt(t)==d**-0.5` and unclamped.

## Complexity
\(O(\text{param groups})\) per step - a constant number of float power evaluations, no state tensors and no history
dependence (the closed form depends only on \(t\)), so resuming from any step is free; \(T\) steps cost \(O(T)\).
