# Solution Notes · Dynatemp Dynamic Temperature

## Using Entropy as a Thermometer

Information entropy $H=-\sum p\ln p$ measures a distribution's uncertainty. Divide it by the **maximum possible entropy** $\ln n$ (the entropy of a uniform distribution)
to normalize it to $C\in[0,1]$, then linearly interpolate into the temperature window:

$$\text{dyn}=T_{\min}+(T_{\max}-T_{\min})\cdot C^{\text{exponent}},\quad
T_{\min}=\max(0,\text{temp}-\Delta),\ T_{\max}=\text{temp}+\Delta$$

- $C\to1$ (nearly uniform, the model is very hesitant) → $\text{dyn}\to T_{\max}$; a high temperature flattens the logits and encourages exploration.
- $C\to0$ (nearly one-hot, the model is already certain) → $\text{dyn}\to T_{\min}$; a low temperature (even approaching 0) preserves that confidence.

> Note this matches intuition: llama.cpp's implementation uses **entropy itself** (not its reciprocal) as the interpolation coefficient, so "more hesitant → hotter."

## exponent Tunes the Curve

$C^{\text{exponent}}$ determines how the temperature moves within the window:

- `exponent = 1`: linear; at 50% entropy the temperature lands exactly in the middle of the window.
- `exponent > 1`: convex, pushing most medium entropies toward $T_{\min}$ (calm by default, heating up only when extremely hesitant).
- `0 < exponent < 1`: concave (e.g. square root), biased toward $T_{\max}$ (hot by default).

## Pitfalls

1. **Natural logarithm**: both the entropy and `max_entropy = -ln(1/n) = ln n` must use `ln`; using `log2` would be off by a constant factor throughout.
2. **`clamp_min` guards against `log(0)`**: when $p_i=0$, $-p_i\ln p_i$ has limit 0; the implementation uses `probs.clamp_min(1e-30)`
   to avoid `0 * -inf = nan`.
3. **`delta <= 0` early exit**: dynamic is off, `dyn` always equals `temp`, and **no** entropy is computed—just
   `softmax(logits/temp)`.
4. Temperature acts in the **logit domain** (`logits/dyn` then softmax), which is equivalent to log-scaling in the probability domain.

## Contract

Consistent with the collection: `filtered` in original order, `chosen` obtained by inverse-transform sampling over the CDF in original order; additionally return the per-row `dyn`
(dynamic temperature), a scalar tensor for a single row and `(...,)` for a batch.
