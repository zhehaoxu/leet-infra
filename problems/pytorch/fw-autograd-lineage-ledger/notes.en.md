# Deep Dive: Autograd Lineage Ledger - Leaves, requires_grad and the Backward Graph

## Sources

On the tutorial side there are the official *Autograd* guide (`docs/source/beginner/basics/autogradqs_tutorial.py`) and *Autograd mechanics* (`docs/source/notes/autograd.rst`): the first covers "leaves, `requires_grad`, `.grad` accumulation, `no_grad`", the second adds "`inference_mode` versus `no_grad`, leaf vs non-leaf, in-place error messages". Both only give conclusion checklists; the per-statement metadata changes (views that borrow a flag, `detach_` being a no-op on inference tensors, which released graphs can be replayed) have to be read off the implementation or measured.

The implementation side is reproduced against torch 2.14.1 (the same locations listed at the top of `solution.py`):

- `Variable::set_history` in `torch/csrc/autograd/variable.cpp`: only grad mode, and only when some operand is a non-inference tensor with `requires_grad`, installs a `grad_fn` - the source of the graph-building rule in rules 1/2.
- `aten/src/ATen/native/TensorViews.cpp`: a view shares `_base` and the base tensor's `requires_grad` (grad mode plays no part), and `detach` creates a new tensor with `_base` cut.
- `run_backward` in `torch/csrc/autograd/engine.cpp`: the implicit gradient only accepts scalar outputs, `retain_graph` decides whether the walked nodes are released, and `AccumulateGrad` in `accumulate_grad.cpp` allocates / accumulates `.grad`.
- `check_inplace` inside the generated in-place wrappers in `VariableType`: the two errors `a leaf Variable that requires grad is being used in an in-place operation` and `a view of a leaf Variable that requires grad...`; `Tensor::detach_` raises `Can't detach views in-place`.
- `Tensor::requires_grad_` rejects `True` on an inference tensor before the "same value is a no-op" short-circuit (`Setting requires_grad=True on inference tensor outside InferenceMode is not allowed`).

## Core and Traps

1. **Inference tensors do not propagate `requires_grad`**: the result of `inference tensor + ordinary leaf` enters the graph driven by the ordinary leaf, while `inference + inference` never enters the graph at all; and an "inference and requires-grad" tensor used as an **operand** triggers `Inference tensors cannot be saved for backward`, a combination excluded by the problem's guarantees.
2. **There are two kinds of views - the biggest trap in the problem**: a view taken in grad mode from a root that wants gradients gets `TBackward0`/`SelectBackward0`/`UnsqueezeBackward0` and **owns metadata**, so after the root's flag is cleared by `detach_` (chain 4, row 2) it still reads `rg=1`; conversely a view taken under `no_grad`/`inference_mode`, or from a root that did not want gradients at the time, **has no metadata**: `is_leaf` is true and the flag is read **live** from the root (the root lights up, it lights up; the root goes dark, it goes dark). `check_borrowed_view_flag` and `check_view_flag_and_detach` pin down each half.
3. **A borrowed-flag view can build a graph but cannot accumulate**: `mul(borrowed, borrowed)` in grad mode reports `rg` true and does install `MulBackward0`, but both input edges are empty - it has no `AccumulateGrad` host. Such a graph backpropagates **successfully while allocating no buffer anywhere**; starting a backward from the view itself (a scalar view) fails outright, because the engine finds neither a node nor a place to accumulate.
4. **`detach_` on an inference tensor is a no-op**: it does not create a tensor but resets autograd metadata, and inference tensors have none to reset, so the flag keeps its value. The only way to clear an inference tensor's flag is `requires_grad_(False)`.
5. **The order of `requires_grad_` checks**: first the inference-and-True rejection (always, even when the flag is already on), then "same value" (a silent no-op, legal on non-leaves and views), and only then "a different value needs a leaf" (non-leaves fail). Chain 6 row 8 is exactly the successful no-op "non-leaf + same value".
6. **A release invalidates buffers, not the graph**: a backward without `retain_graph` releases every node it walks, but only nodes that **re-read saved tensors** in backward (mul, sigmoid, relu) raise `Trying to backward through the graph a second time` on a second pass; add/neg/sum/mean/clone and all three views do not re-read data, so the release does not affect them at all. `check_non_saving_nodes_replay` pins this with `leaf -> sum` backpropagated three times (including a second pass without retain), while `check_backward_graph_consumption` pins the other half with a shared `MulBackward0`.
7. **The identity of a gradient buffer**: `grad_id` records the statement index of the **first `backward` that allocated it**; accumulating into the same buffer again keeps it; after `grad_none` the next backward **re-allocates** (a new statement index); `grad_zero_` only zeroes and keeps the identity. Chain 2 strings these four stages together.
8. **`retain_grad` only decides that an intermediate also keeps a buffer**: leaves always do; calling it on a tensor that currently wants no gradients fails. Note that "marked" does not mean "will get one": if the backward after the mark fails because the graph was released, that intermediate's `.grad` stays empty forever (`check_diamond_allocates_one_buffer`, row 9).
9. **An `add_` node keeps the subject's previous node as an input edge**: so leaves reached through the old node before the write still receive gradients on the new graph (chain 10 row 6, the second leaf), and the subject's `grad_fn` is replaced, which is what `probe` reports.
10. **A scalar leaf can be backpropagated directly**: a scalar without `grad_fn` that owns its metadata (a plain leaf, or the host of a scalar buffer lit by `requires_grad_`) succeeds and allocates its own buffer, whereas a scalar view that borrows its flag fails. The two differ only in `ok`, so it is easy to assume "no `grad_fn` means no backward".

## Real Behaviour Outside the Domain (Deliberately Excluded)

- **Changing `requires_grad_` on a borrowed-flag view**: torch then creates metadata for it on the fly and the view and the root diverge from that point on. The behaviour depends on internal timing, so the problem excludes it with a guarantee (the subject of `requires_grad_` is never a view).
- **An "inference and requires-grad" tensor as an operand**: raises `Inference tensors cannot be saved for backward`, which breaks the build/no-build dichotomy, so it is excluded.
- **Non-scalar `backward()`**: the implicit gradient without a `gradient` argument is only legal for scalar outputs; other cases raise `grad can be implicitly created only for scalar outputs`. The problem only fails on `rank > 0` and does not cover an explicit `gradient` argument.
- **A backward that dies halfway**: on a multi-branch graph the real engine may allocate a leaf's buffer first and then hit a released node and fail. That would break "a failure changes nothing", so the oracle's `partial backward` guard keeps such statements out of the generator and the program guarantees state it explicitly.
- **Higher order**: `create_graph=True`, differentiating a `grad`, `register_hook`, `torch.autograd.grad`, `functional_call` and friends are out of scope; `torch.compile` and anything CUDA/device-related is excluded by the collection-wide red line.
- The op set has no `exp`/`mm`/`sub`/`div`: it only picks representatives for the four classes "builds a graph / does not build a graph / re-reads saved tensors / does not", so the problem does not degenerate into memorizing class names.

## Verification

`solution` is a pure-Python metadata simulation and does not import torch. `test_cases.py` uses torch as the oracle:

- `Runner` executes the real torch statements one by one and keeps the tensor created by statement `i` in `ts[i]`; `view_of` is resolved by looking up `id(t._base)` among the statement indices and `grad_id` by looking up `id(t.grad)` among the `backward` statements that allocated it; a `keep` list holds extra references so that `id` reuse cannot skew the identity test.
- The observations read only six public surfaces (`requires_grad`/`is_leaf`/`torch.is_inference`/`dim`/`grad_fn`/`_base`/`grad`) and never touch values, so the conclusions stay stable across torch versions as long as the semantics hold.
- The generator proposes a candidate statement and lets real torch try it: an error outside the whitelist (`EXPECTED_ERRORS`) means the statement hit an excluded combination and the whole statement is dropped (`OutOfDomain`); a whitelisted error is additionally checked for "did it allocate a buffer halfway", and dropped as well if so.
- Random cross-checking: 8 fixed seeds x 40 programs of 6-26 statements each, 5182 statements in total (754 of them rejected/failed rows), with the solution equal to torch on every row and zero divergence; all randomness comes from `random.Random(seed)` with no global state.
- The 6 example chains in the statement were measured in both directions, by the reference implementation and by torch; 13 targeted checks (graph consumption, replay, diamond buffer, in-place rewrite, regions and lineage, leaf flags and buffers, borrowed views, views and `detach_`, `requires_grad_`/`retain_grad`/scalar leaf, inference and `detach_`, invalid input) are each cross-checked against torch.

## Complexity

Building a graph or walking a backward pass is O(graph size) per statement, so a whole program is O(statements x graph size); with at most 40 statements the full cross-check (nearly 5200 statements) finishes within a second on this machine.
