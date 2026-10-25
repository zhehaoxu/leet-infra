# Walkthrough: Gradient Accumulation + Global Norm Clipping + Momentum SGD

## Source and verification
- **The clipping convention** comes from the official PyTorch `torch.nn.utils.clip_grad_norm_`. Verified against its source `clip_grads_with_norm_` / `get_total_norm`: the per-parameter \(\|g_p\|_2\) values are stacked and `vector_norm(norm_type=2)` is taken, i.e. \(\text{total}=\sqrt{\sum_p\|g_p\|_2^2}\); the coefficient is `clip_coef = max_norm / (total_norm + 1e-6)`, then `clamp_(max=1.0)`, and finally every gradient gets `mul_(clip_coef)`. This problem **first averages the accumulated gradient** and only then applies that clipping, matching the mainstream framework practice of "divide by the accumulation steps after accumulating, then clip".
- **Gradient accumulation** is a standard trick for large-batch training (`accum_steps` micro-batches folded into one update), not specific to any single paper.
- **Momentum SGD**: Polyak's momentum method (1964); PyTorch `SGD(momentum=...)` uses \(b\leftarrow\mu b+g,\ p\leftarrow p-lr\cdot b\) (the buffer carries no \((1-\mu)\) factor), and this problem keeps that convention.

## Core flow (every pitfall is about ordering)
1. **Accumulate vs update timing**: `step()` adds to \(a\) once per micro-batch, but \(p\) is actually updated only when \(m\) reaches `accum_steps`. On the early returns you must **never** touch \(p\) and must never clear \(a\).
2. **Average before clipping**: the global norm uses the norm of \(\bar g=a/\text{accum}\), not the norm of \(a\). Forget the division by `accum_steps` and the clipping coefficient is off by a factor of `accum_steps`.
3. **Global (cross-parameter) norm**: the coefficient \(c\) is one scalar shared by **all** participating parameters (sum the squared per-parameter norms, then take the square root), not a per-parameter clip.
4. **`+1e-6` and `clamp(max=1)`**: the denominator gets \(10^{-6}\) to guard against division by zero; the cap of 1 on \(c\) guarantees shrink-only when the threshold is exceeded and never amplification.
5. **Reset**: after one effective update, zero \(a\) and \(m\); but the momentum buffer \(b\) is **not** cleared and keeps accumulating across large batches.

## Comparison setup
The judge recomputes independently with `_oracle_clipaccum` (a list-based version that writes the clipping coefficient as a Python `float` `min(...)` instead of a tensor `clamp`, so its structure differs from the reference solution), compared under `allclose(atol=1e-5, rtol=1e-5)`; measured runs are **exactly identical** elementwise (maxdiff=0). Five checks:
- `check_example`: the concrete parameter values with `accum=4` and large `randn*2.0` gradients that trigger clipping, after 12 micro-batches (3 effective updates).
- `check_matches_oracle_clipped`: three mixed shapes, `accum=3`, `momentum=0.9`, `randn*3.0`, cross-checked against `_oracle_clipaccum`.
- `check_partial_micro_no_update`: parameters are completely unchanged when only 2 micro-batches are fed (`accum=3`), verifying "no update and no accumulator reset on an early return".
- `check_none_grad_param_skipped`: a parameter that never has a gradient gets no state created and keeps its value.
- `check_returns_optimizer_instance`: `solve` returns a `torch.optim.Optimizer` instance.

## Complexity
Per micro-batch \(O(\text{numel})\) for accumulation; per large batch an extra \(O(\text{numel})\) for the norm and the update. State \(a,b\) (two tensors, same footprint as momentum SGD).
