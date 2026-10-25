# Parameter Freezing, Head Swap and Optimizer Groups

## Problem

The standard fine-tuning recipe: freeze the whole backbone, drop the old head, attach a new head, and hand the optimizer only the trainable parameters. Every step produces framework-level observable accounting: parameter-count changes, `requires_grad` flags, optimizer state entries, and frozen parameters that must remain bit-identical after `step()`. This problem walks the full flow and tallies the books.

## Rules

Implement `solve()` and return the behavior-ledger dict following the fixed flow. The model structure, the explicit weight formula \(\texttt{\_explicit(shape,start)}=(\texttt{arange}(n)+\texttt{start})/100\), the 8-step procedure and the return format are all specified in the starter docstring, the single source of truth. Key points:

1. Freeze with `requires_grad_(False)`; after swapping, the new `nn.Linear` is trainable by default, and the dimensions differ (a \(6\to4\) head becomes \(6\to5\)), so the total parameter count changes.
2. The optimizer collects only parameters with `p.requires_grad` true; run `SGD(lr=0.1, momentum=0.9)` for 3 steps with the MSE loss \(\frac{1}{N}\sum (f(x)-y)^2\).
3. Frozen parameters must be **bit-identical** (`torch.equal`) to their pre-loop clones after the 3 steps, and their `.grad` must stay `None` after backward (autograd never computes gradients for them at all).
4. `len(opt.state)` counts only the parameters that actually got a momentum buffer.

## Examples

```
Input: none
Output (excerpt of the full reference run):
{
 "total_before": 124, "total_after": 131,
 "trainable_before": 0, "trainable_after": 35,
 "flags": [["encoder.0.weight", false], ["encoder.0.bias", false],
           ["encoder.2.weight", false], ["encoder.2.bias", false],
           ["head.weight", true], ["head.bias", true]],
 "losses": [6.816761016845703, 11.316238403320312, 1.1022732257843018],
 "final_head_weight_flat": [0.002984001999720931, 0.013380030170083046,
   0.02377604879438877, 0.03417206183075905, 0.04456809163093567,
   0.054964106529951096, ...],
 "final_head_bias_flat": [-0.006162865553051233, -9.81296252575703e-05, ...],
 "frozen_bitwise_identical": true, "frozen_grad_is_none": true,
 "n_opt_state_params": 2
}
```

The loss rebound from 6.82 to 11.32 at step 1 is momentum overshoot; step 2 drops back to 1.10.

## Constraints

- The whole flow is deterministic: initial weights / inputs / targets all come from `arange` formulas; randomness is forbidden.
- Numeric comparison uses `atol=1e-5, rtol=1e-5`; counts / booleans / flags must match exactly.
- The order of `flags` must match `named_parameters()`.
