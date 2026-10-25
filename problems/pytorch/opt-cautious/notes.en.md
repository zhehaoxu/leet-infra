# Walkthrough: Cautious Sign-Consistency Mask (Cautious Optimizer)

## Source
Liang, Zhang, Kwon, et al., *Cautious Optimizers: Improving Training with One Line of Code* (arXiv:2411.16085). The masking operator matches `apply_cautious(update, grad)` in the kozistr/pytorch-optimizer base class exactly:

```
mask = (update * grad > 0).to(grad.dtype)
mask.mul_(mask.numel() / (mask.sum() + 1))
update.mul_(mask)
```

## Intuition
A momentum-style update \(u_t\) is an EMA of past gradients; on a few coordinates it can have the **opposite sign** of the latest gradient \(g_t\) — that amounts to "walking backwards against the current gradient". The authors measured that such conflicting coordinates make up a substantial fraction, and suppressing them stabilizes and speeds up training. The recipe is:

1. keep only the coordinates with \(u_t\odot g_t>0\) (same sign as the current gradient) and set the rest to 0;
2. since zeroing weakens the overall magnitude, rescale by \(\dfrac{N}{\sum M+1}\) (\(N\) is the element count, \(\sum M\) the number of surviving coordinates; the \(+1\) prevents division by zero when every coordinate conflicts), so the kept directions "give back" the magnitude that was removed.

For AdamW, \(u\) is the **update direction before multiplying `lr`**, \(\hat m/\text{denom}\); the mask acts on it, and only at the end \(p\leftarrow p-lr\cdot(u\odot M)\). This problem builds on AdamW (decoupled weight decay, with bias correction), aligned with arXiv:1711.05101.

## Pitfalls for matching the reference solution
1. **The `+1` belongs in the denominator**: the scaling factor is \(N/(\sum M+1)\), not \(N/\sum M\).
2. **The mask uses the raw gradient \(g\)** (not \(m\)): the criterion is \(u\odot g>0\). Decoupled weight decay only acts on \(p\) and leaves \(g\) unchanged, so \(g\) is just `p.grad`.
3. **\(N\)** is the number of elements of the current parameter tensor (`mask.numel()`), computed per parameter.

## Comparison setup
The judge recomputes everything independently with a vectorized replay (`_oracle_cautious`) using the same AdamW direction + mask; it also runs `check_cautious_mask_changes_result`, which compares against "AdamW without the mask" and asserts that the mask really changes the result (i.e. that conflicting coordinates were indeed suppressed). Parameters with `grad is None` are skipped, weight decay included.

## Complexity
Same as AdamW (\(O(\text{numel})\), two state tensors), plus one extra elementwise comparison and sum.
