# Autograd Lineage Ledger: Leaves, requires_grad and the Backward Graph

## Problem

All of PyTorch autograd's "magic" lives in one metadata table. A tensor is either a **leaf** (`grad_fn is None` - the thing the user created and `optimizer.step()` actually updates) or it carries a **backward node** (`grad_fn`, pointing at the op that produced it). `requires_grad` decides whether a backward pass is worth running at all. `.backward()` walks `next_functions` and pushes gradients into the leaves' `.grad` buffers. `no_grad` and `inference_mode` both mean "do not build a graph", yet they differ in what they do to an existing `requires_grad`, to in-place writes, and to `detach_`. The official *Autograd* tutorial and the *Autograd mechanics* notes only state the conclusions; how each statement changes the metadata is something you learn by experimenting.

This problem compresses that knowledge into a small **statement language**: a program runs a sequence of statements, each either creating a new tensor or mutating / reading an existing one; tensors are addressed by statement index. For every statement you report an observation row - `requires_grad`, `is_leaf`, `is_inference`, rank, the `grad_fn` class name, the root of a view, whether `.grad` exists, and which `backward` statement allocated that buffer. No tensor values are involved; everything is metadata bookkeeping.

## Rules

Implement `solve` (the statement table, the nine observation columns and the invalid-input contract are in the starter docstring). The essentials:

1. **A leaf and a node are opposites**: `leaf == (node is None)`. A value/view op gets a node only in grad mode and only when some operand currently has `requires_grad` set and is not an inference tensor (`AddBackward0`, `MulBackward0`, `SumBackward0`, and `TBackward0`/`SelectBackward0`/`UnsqueezeBackward0` for views, ...); results built under `no_grad`/`inference_mode` never enter the graph. An inference tensor's own `requires_grad` never propagates a graph.
2. **Views and `detach`**: a view copies `requires_grad` and `infer` from its operand **regardless of `ctx`** (a view taken under `no_grad` still wants gradients but, having no `grad_fn`, is a leaf), and `view_of` points at the **root** tensor; `detach` always reports `requires_grad=False` and is not a view. A view that did not get a node (its root did not want gradients then, or it was built under `no_grad`/`inference_mode`) holds **no metadata of its own**: its flags are read **live** from the root afterwards.
3. **In-place writes**: `add_` drags the subject into the graph and swaps its node for `AddBackward0` (a former leaf stops being a leaf). Writing into "a leaf that wants gradients" or "a view of such a leaf" is rejected. `detach_` is rejected on views; an inference tensor has no autograd metadata to reset, so an in-place `detach_` cannot clear its flag.
4. **`requires_grad_`**: writing the current value is a silent no-op; a different value is legal only on leaves; on an inference tensor it can only turn the flag off (turning it on is rejected even when it is already on).
5. **`backward`**: the implicit gradient only works for a scalar output; the subject must currently want gradients, and a view that merely borrows the flag does not qualify (the engine finds no node and nowhere to accumulate). A backward pass releases the graph it walked, but only nodes that **re-read saved tensors** in backward (mul/sigmoid/relu) block a second pass; sum/add/neg/clone/views do not, so the same graph can be replayed. Gradients land on leaf references and on `retain_grad`-ed intermediates: an empty `.grad` gets a new buffer (`grad_id` records the statement index of that backward), an existing one is accumulated in place; after `grad_none` the next backward allocates again.
6. **Reads**: `grad` reports the `.grad` buffer itself as a new tensor (fails when there is none), `grad_none` clears it, `grad_zero_` zeroes it in place without changing its identity, `probe` just re-reads a row.

## Examples

Each line is `statement index  statement -> observation`; `rg/leaf/infer/rank` are the four metadata fields, `node` is the `grad_fn` class name, `base` is the root index of a view, `grad=#k` means that buffer was allocated by statement `k` (a backward pass), and `fail` means torch rejected the statement and the program state did not change.

Chain 1: a backward pass consumes the graph; a second pass fails whether or not `retain_graph` is requested, and the leaf's buffer is allocated only once.

```
 0 leaf(True, grad)   -> rg=1 leaf=1 infer=0 rank=2
 1 mul(0, 0, grad)    -> rg=1 leaf=0 infer=0 rank=2 node=MulBackward0
 2 sum(1, grad)       -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
 3 backward(2, False) -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
 4 backward(2, False) -> fail
 5 backward(2, True)  -> fail
 6 probe(0)           -> rg=1 leaf=1 infer=0 rank=2 grad=#3
```

Chain 2: the same graph can be backpropagated three times with `retain_graph=True`, accumulating into the **same** buffer (`grad=#3` never changes); only after `grad_none` does the next pass hand out a new one (`grad=#9`), while `grad_zero_` zeroes in place.

```
 0 leaf(True, grad)  -> rg=1 leaf=1 infer=0 rank=2
 1 mul(0, 0, grad)   -> rg=1 leaf=0 infer=0 rank=2 node=MulBackward0
 2 sum(1, grad)      -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
 3 backward(2, True) -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
 4 probe(0)          -> rg=1 leaf=1 infer=0 rank=2 grad=#3
 5 backward(2, True) -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
 6 probe(0)          -> rg=1 leaf=1 infer=0 rank=2 grad=#3
 7 grad_none(0)      -> rg=1 leaf=1 infer=0 rank=2
 8 probe(0)          -> rg=1 leaf=1 infer=0 rank=2
 9 backward(2, True) -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
10 probe(0)          -> rg=1 leaf=1 infer=0 rank=2 grad=#9
11 grad_zero_(0)     -> rg=1 leaf=1 infer=0 rank=2 grad=#9
12 probe(0)          -> rg=1 leaf=1 infer=0 rank=2 grad=#9
```

Chain 3: tensors built under `no_grad` and `inference_mode` are leaves that want no gradients; `inference_mode` may create a leaf that *does* want gradients, which still cannot be backpropagated (no graph), and `detach` of it keeps the inference identity.

```
 0 leaf(True, grad)           -> rg=1 leaf=1 infer=0 rank=2
 1 mul(0, 0, no_grad)         -> rg=0 leaf=1 infer=0 rank=2
 2 mul(0, 0, inference_mode)  -> rg=0 leaf=1 infer=1 rank=2
 3 leaf(True, inference_mode) -> rg=1 leaf=1 infer=1 rank=2
 4 detach(3, grad)            -> rg=0 leaf=1 infer=1 rank=2
 5 backward(3, False)         -> fail
 6 neg(3, inference_mode)     -> rg=0 leaf=1 infer=1 rank=2
 7 probe(4)                   -> rg=0 leaf=1 infer=1 rank=2
```

Chain 4: a view taken under `no_grad` gets no node yet borrows the root's flag; `add_` may not write a "view of a leaf that wants gradients", `detach_` may not write a view, but writing a plain leaf is fine - and its already-allocated buffer survives the flag being cleared.

```
 0 leaf(True, grad)   -> rg=1 leaf=1 infer=0 rank=2
 1 t(0, grad)         -> rg=1 leaf=0 infer=0 rank=2 node=TBackward0 base=0
 2 index(1, grad)     -> rg=1 leaf=0 infer=0 rank=1 node=SelectBackward0 base=0
 3 sum(2, grad)       -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
 4 backward(3, False) -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
 5 t(0, no_grad)      -> rg=1 leaf=1 infer=0 rank=2 base=0
 6 index(5, no_grad)  -> rg=1 leaf=1 infer=0 rank=1 base=0
 7 add_(5, 5)         -> fail
 8 add_(2, 2)         -> fail
 9 detach_(1)         -> fail
10 detach_(0)         -> rg=0 leaf=1 infer=0 rank=2 grad=#4
```

Chain 6: `retain_grad` keeps a buffer on an intermediate; writing the current value of `requires_grad` on a non-leaf is a no-op; clearing a leaf's flag leaves the buffer alone.

```
 0 leaf(False, grad)        -> rg=0 leaf=1 infer=0 rank=2
 1 requires_grad_(0, True)  -> rg=1 leaf=1 infer=0 rank=2
 2 mul(0, 0, grad)          -> rg=1 leaf=0 infer=0 rank=2 node=MulBackward0
 3 retain_grad(2)           -> rg=1 leaf=0 infer=0 rank=2 node=MulBackward0
 4 sum(2, grad)             -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
 5 backward(4, False)       -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
 6 probe(2)                 -> rg=1 leaf=0 infer=0 rank=2 node=MulBackward0 grad=#5
 7 grad(2)                  -> rg=0 leaf=1 infer=0 rank=2
 8 requires_grad_(2, True)  -> rg=1 leaf=0 infer=0 rank=2 node=MulBackward0 grad=#5
 9 probe(0)                 -> rg=1 leaf=1 infer=0 rank=2 grad=#5
10 requires_grad_(0, False) -> rg=0 leaf=1 infer=0 rank=2 grad=#5
11 probe(0)                 -> rg=0 leaf=1 infer=0 rank=2 grad=#5
```

Chain 10: `add_` keeps the old node as an input edge of the new one, so the second leaf gets a gradient too; after the write the graph is a fresh one.

```
 0 leaf(True, grad)   -> rg=1 leaf=1 infer=0 rank=2
 1 leaf(True, grad)   -> rg=1 leaf=1 infer=0 rank=2
 2 mul(0, 0, grad)    -> rg=1 leaf=0 infer=0 rank=2 node=MulBackward0
 3 add_(2, 1)         -> rg=1 leaf=0 infer=0 rank=2 node=AddBackward0
 4 sum(2, grad)       -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
 5 backward(4, True)  -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
 6 probe(1)           -> rg=1 leaf=1 infer=0 rank=2 grad=#5
 7 add_(2, 0)         -> rg=1 leaf=0 infer=0 rank=2 node=AddBackward0
 8 sum(2, grad)       -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
 9 backward(8, False) -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
10 probe(2)           -> rg=1 leaf=0 infer=0 rank=2 node=AddBackward0
```

Walking chain 4 rows 5-6: the view produced by `t(0, no_grad)` has no `grad_fn` and is thus a leaf; its `requires_grad` is borrowed - the root currently wants gradients, so it reads as 1. Row 7 `add_(5, 5)` therefore hits "a view of a leaf Variable cannot be written in place"; row 10 `detach_(0)` acts on a plain leaf, clearing the flag while keeping the buffer allocated by the backward in row 4.

## Constraints

- At most 40 statements; tensor indices start at 0 and only refer to statements that successfully created a tensor.
- All observation columns are booleans, integers or non-empty strings and are compared exactly; no floating point, no randomness.
- Do not actually run torch (or numpy) on the statements and read the metadata back to look the answer up.
- Invalid input (not a list, malformed statement, out-of-range index, `flag` not a bool, ...) raises `ValueError`, which is different from "torch rejected the statement"; the latter is reported as a failed observation row.
- The guarantees are listed in the starter docstring: operand shapes always broadcast, an "inference and requires_grad" tensor never appears as an operand, the subject of `requires_grad_` is never a view, and a failing backward never leaves gradients allocated halfway.
