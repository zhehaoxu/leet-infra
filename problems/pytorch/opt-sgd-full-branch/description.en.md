# SGD Full-Branch Update Ledger

`torch.optim.SGD` looks like a single update rule, but four knobs interfere with each
other: `nesterov` and `dampening` are mutually exclusive, `dampening` is **ignored on the
very first momentum step**, `maximize` negates the gradient **before** `weight_decay`, and
`dampening` is silently dropped when `momentum == 0`. The docstring pseudo-code only shows
the trunk; these branches live in the source.

This problem mirrors torch 2.14.1's `torch/optim/sgd.py` line by line: the four constructor
checks in `SGD.__init__` plus the single-step path of `_single_tensor_sgd`. It runs several
steps at once and returns a ledger you can compare step by step.

Implement `solve(params, grads, lr, momentum=0.0, dampening=0.0, weight_decay=0.0,
nesterov=False, maximize=False, buffers=None)`.

## Inputs

- `params`: a tensor of any shape (float32 / float64).
- `grads`: a sequence of length `T`; each item is a tensor shaped like `params`, or `None`
  (meaning **no gradient** for that step).
- `lr` / `momentum` / `dampening` / `weight_decay`: scalars.
- `nesterov` / `maximize`: booleans.
- `buffers`: the momentum buffer **already present** when the call starts; `None` means no
  history.

## Validation

Four gates, raising `ValueError` in this **order**, with the upstream messages (`{}` is the
argument value):

```
Invalid learning rate: {}
Invalid momentum value: {}
Invalid weight_decay value: {}
Nesterov momentum requires a momentum and zero dampening
```

The order is `lr` → `momentum` → `weight_decay` → the `nesterov` condition, and that
condition is `momentum <= 0 or dampening != 0`. `dampening` itself is **never range
checked** (it may exceed 1 or be negative), and with `momentum == 0` a non-zero `dampening`
does not raise either.

## One step

For step `t` (`t = 0, 1, …`):

1. `grads[t] is None`: **skip the whole step** — no buffer is created, the parameters do not
   move, and `dirs[t]` is `None`.
2. `maximize` is true: `g ← -grads[t]` (**negation happens before weight_decay**).
3. `weight_decay != 0`: `g ← g + weight_decay · params`, where `params` is the value **before**
   this step's update.
4. When `momentum != 0`:
   - no buffer yet (`buffers is None` and none created so far): `buf ← g` (a copy), and
     **`dampening` is ignored on this step**;
   - a buffer exists (including an explicitly passed **all-zero** buffer):
     `buf ← momentum · buf + (1 - dampening) · g`;
   - `nesterov` is true: `step ← g + momentum · buf`, using the **just updated** `buf`;
     otherwise `step ← buf`.
5. `momentum == 0`: `step ← g`, the buffer stays all zeros, and `dampening` plays no part.
6. `params ← params - lr · step`. With `lr == 0` the parameters stay put but the buffer
   **keeps advancing**.

## Returns

One dict with four keys:

- `params`: the updated parameters.
- `buffers`: the final momentum buffer; all zeros when `momentum == 0` or when it was never
  created.
- `path`: `T + 1` parameter snapshots; `path[0]` is a copy of the input.
- `dirs`: length `T`, holding the step direction actually applied at each step (`None` for
  skipped steps).

Inputs are never modified in place; the four returned fields are independent, so mutating
`dirs` in place cannot touch `buffers`.

## Constraints

- `T` may be 0: `path` then has a single entry, `params` is unchanged, and `buffers` is all
  zeros (or a copy of the argument).
- Step-by-step agreement with `torch.optim.SGD` at `1e-12` in float64.
- No in-place modification of `params` / `grads` / `buffers`.

## Examples

```
Input: params=[1.0, -2.0], grads=[[0.5, -1.0], [1.0, 1.0]], lr=0.1, momentum=0.9, dampening=0.3
Output: params=[0.835, -1.88], buffers=[1.15, -0.2]
```

The first gradient creates the buffer `[0.5, -1.0]`; the second one goes through
`0.9·buf + 0.7·g` and yields `[1.15, -0.2]`.

```
Input: params=[1.0, 1.0], grads=[[1.0, -2.0]], lr=0.1, momentum=0.9, dampening=0.5
Output: dirs=[[1.0, -2.0]], buffers=[1.0, -2.0]
```

`dampening=0.5` **does not show up** on this step — the first momentum buffer is the gradient
itself; pass `buffers=[[0.0, 0.0]]` instead and the direction becomes `[0.5, -1.0]`.
