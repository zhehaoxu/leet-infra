# Dynamic Temperature Sampling (Dynatemp)

## Problem

A fixed temperature is either always "calm" or always "unleashed", but a model's confidence **changes step by step**.
**Dynatemp** (llama.cpp's `--dynatemp-range/-exp`) lets the temperature slide within the
$[\,\text{temp}-\Delta,\ \text{temp}+\Delta\,]$ window according to the distribution's **entropy** (uncertainty): the **flatter** the distribution (higher entropy, the more hesitant the model)
→ the **higher** the temperature (reshuffling more decisively); the **sharper** the distribution (low entropy, the model is already confident) → the temperature is **closer to the lower bound**.
The `exponent` controls the bending direction of this interpolation curve.

## Rules

- Implement `solve(logits, temp, delta, exponent, u)`: `logits` has shape `(..., V)`; `temp > 0`,
  `delta ≥ 0`, `exponent > 0` are scalars; `u` takes values in $[0,1)$ (shape `(...,)` or scalar).
- Steps (per row, with $n=V$):
  1. `probs = softmax(logits)`.
  2. If `delta <= 0`: **disable dynamic**, set $\text{dyn}=\text{temp}$, `filtered = softmax(logits/temp)`.
  3. Otherwise:
     - $\text{max\_entropy} = -\ln(1/n) = \ln n$;
       $\text{entropy} = \sum_i -p_i \ln p_i$;
       $C = \text{entropy}/\text{max\_entropy} \in [0,1]$.
     - $T_{\min} = \max(0,\ \text{temp}-\Delta)$, $T_{\max} = \text{temp}+\Delta$;
       $$\text{dyn} = T_{\min} + (T_{\max}-T_{\min})\cdot C^{\text{exponent}}$$
     - `filtered = softmax(logits / dyn)`.
  4. Use `u` to inverse-transform sample on the CDF of `filtered` (in original order) to obtain `chosen`.
- Return `(filtered, chosen, dyn)`: `filtered` has the same shape as `probs`, and `dyn` has shape `(...,)` (the dynamic temperature of each row).

## Examples

```
Input: logits = [2, 1, 0.5, -1, -2, -3], temp = 1.0, delta = 0.3, exponent = 1.0, u = 0.5
  probs = [0.600294, 0.220836, 0.133944, 0.029887, 0.010995, 0.004045]
  entropy = 1.085951, max_entropy = ln 6 = 1.791759, C = 0.606081
  T_min = 0.7, T_max = 1.3 → dyn = 0.7 + 0.6 * 0.606081^1 = 1.063648
  filtered = softmax(logits / 1.063648) = [0.579178, 0.226207, 0.141368, 0.034506, 0.013477, 0.005264]
  CDF = [0.579178, ...], u = 0.5 → the first entry is already ≥0.5 → chosen = 0
Output: filtered = [0.579178, 0.226207, 0.141368, 0.034506, 0.013477, 0.005264], chosen = 0, dyn = 1.063648
```

## Constraints

- float32 input; `temp > 0`, `delta ≥ 0`, `exponent > 0`; `u ∈ [0,1)`
- Each row of `filtered` sums to 1 (softmax removes no candidates); `chosen` is the position hit by inverse-transform sampling
- `entropy` uses the **natural logarithm**; when `delta <= 0`, `dyn` always equals `temp`
- Uniform distribution → $C=1$ → `dyn=T_max`; extremely peaked distribution → $C\to0$ → `dyn\to T_min`
