# Autograd Graph Mechanics Ledger: Leaves, detach, Disable Modes and Grad Accumulation

## Problem

The autograd dynamic graph is made of "leaves → ops → grad_fn nodes". Writing training code forces four questions: which tensors are leaves? What does `detach` actually share? What is the essential difference between `no_grad` and `inference_mode` outputs? Do two `backward` calls overwrite or accumulate? This problem turns those four questions into a ledger you must reconcile item by item.

## Rules

Implement `solve(x)` (`x`: a 1-D floating-point tensor with `requires_grad=False`). Build the graph in 12 prescribed steps and return the ledger dict (field meanings in the starter comments):

| Field | Anchor (`x=[1.0, 2.0]`) | Tested concept |
|---|---|---|
| `w_is_leaf` / `w_copies_input` | `True` / `True` | the leaf must be built via `clone`, not sharing input storage |
| `y_is_leaf` / `y_grad_fn` | `False` / `"AddBackward0"` | op outputs are non-leaves; grad_fn is the last op |
| `chain` | `["SumBackward0", "AddBackward0", "MulBackward0", "AccumulateGrad"]` | node chain from output to leaf; `+1` sits between mul and sum |
| `leaf_accumulator` | `"AccumulateGrad"` | node class on the graph's input side |
| `d_is_leaf` / `d_requires_grad` / `d_shared_storage` | `True` / `False` / `True` | detach = a new leaf over the same storage |
| `n_requires_grad` | `False` | `no_grad` outputs are non-differentiable |
| `t_requires_grad` / `t_is_inference` / `t_rg_error` | `False` / `True` / `True` | inference tensors cannot gain grad afterwards |
| `w_grad_first` / `w_grad_second` | `[2,2]` / `[4,4]` | gradients accumulate instead of overwriting |
| `d_grad` | `[3,3]` | the detached tensor can be re-rooted as a leaf of a second graph |

Requirements:

1. The input must stay untouched: same values, `requires_grad` still `False`.
2. Raise `ValueError` when `x` is not a 1-D floating-point tensor.
3. The step-9 `backward` must pass `retain_graph=True`; step 10 must not.

## Example

```
Input: x = [1.0, 2.0]
y = w*2+1 = [3, 5]
w.grad: [2,2] after the first backward, [4,4] after the second   # 2*ones accumulated
d.grad = [3, 3]                                                  # second graph (d*3).sum()
```

## Constraints

- Any 1-D shape, float32/float64; the gradient ledger is value-independent (`dy/dw=2` is constant)
- The judge randomizes the input: `requires_grad`, values and `_version` must all stay unchanged
