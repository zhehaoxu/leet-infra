# Layout Ledger: shape/stride/offset and the view-or-copy Decision

## Problem

A PyTorch tensor is not "a chunk of data" but "one storage plus three pieces of metadata": `shape`, `stride` and `storage_offset`. The official *View operations* guide splits the ops into families - always a view (`transpose`/`permute`/`narrow`/`select`/slicing/`unsqueeze`/`squeeze`/`expand`/`view`), view-or-new-tensor (`reshape`/`flatten`) and always a copy (`contiguous` when needed). Why `t.expand(3,4).reshape(12)` has to copy while `t.permute(2,0,1).reshape(4,6)` does not is the densest cluster of "guess it if you cannot read it" material in the beginner tutorials, and it is the direct explanation for doubled memory, an unexpected `is_contiguous() == False`, and `view()` failing with `at least one dimension spans across two contiguous subspaces`.

Given the metadata of a base tensor and a sequence of view operations, compute **step by step** the resulting `shape`, `strides`, `storage_offset`, `is_contiguous()` and whether that step is a view (`view`) or a copy (`copy`). No tensor values are involved.

## Rules

Implement `solve` (argument shapes, the op set and the invalid-input contract are in the starter docstring). Negative dimensions are always normalized with `+ ndim` first.

1. `transpose`/`permute` only reorder the `shape` and `strides` components; `offset` is untouched.
2. `unsqueeze(d)` inserts a length-1 dimension whose stride is `size[d] * stride[d]`; when `d == ndim` (appended last) the stride is 1. So inserting into `(3,4)/(4,1)` at positions 0/1/2 yields `(12,4,1)`, `(4,4,1)`, `(4,1,1)`.
3. `squeeze` merely deletes components and never recomputes the strides that stay; naming a dimension whose length is not 1 makes the whole step a no-op (same for a 0-dim tensor).
4. `narrow`/`select`/`slice` all advance the offset: `offset += start_norm * stride[d]`, where `start_norm` first gets `+= size[d]` (if negative) and is then clamped into `[0, size[d]]`; for `slice` the end also gets `+= size[d]` and is clamped into `[start_norm, size[d]]`, the length is `ceil((end - start_norm)/step)` and that dimension's stride is multiplied by `step` - a zero-length result still gets its stride from the same formula and the offset still moves to `start_norm`. `select` is a length-1 take followed by dropping the dimension (which is why selecting an inner dimension often leaves a non-contiguous tensor).
5. `expand(sizes)` right-aligns and goes dimension by dimension: equal lengths (including 1 vs 1) keep the old stride; original length 1 with a larger target becomes 0; anything else is invalid. Extra leading dimensions get stride 0, except when their length is 1, in which case the stride is `size[i+1] * stride[i+1]` (0 if it is already the last dimension).
6. The view test for `reshape`/`flatten` (ATen `computeStride`): cut the old shape from the innermost dimension outwards into **contiguous subspaces** - with `tensor_numel` the product of the lengths accumulated inside the current chunk and `base` the chunk's base stride, dimensions `i` and `i+1` share a chunk iff `size[i] == 1` or `stride[i] == tensor_numel(from i+1 on) * base`. Each chunk must be consumed exactly by the target shape, walking the target from its innermost dimension outwards; the j-th target dimension inside a chunk gets stride = (product of the lengths already consumed in that chunk) x `base`, and a length-1 target dimension consumes nothing but still takes its stride from the same running product. If some chunk is not consumed exactly, or target dimensions are left over, the step is a `copy`. After a copy the strides are the standard contiguous strides of the target shape (lengths <= 1 do not advance the running product) and the offset becomes 0. When the total length is 0 a `reshape` is always a view: the strides are copied verbatim if the shape is unchanged, otherwise they are the standard contiguous strides; a `reshape` of a 0-dim base always yields all-1 strides. `flatten(a,b)`: `a == b` is a no-op, on a 0-dim tensor it is a reshape to a single length-1 dimension, otherwise the merged shape goes through the test above.
7. `contiguous` is a `view` (metadata unchanged) when the current layout is already C-contiguous, otherwise a `copy`. C-contiguity test: always true for `ndim == 0` or total length 0; otherwise walk outwards from the innermost dimension with the expected stride starting at 1, requiring `stride == expected` or `size == 1` per dimension, then `expected *= size`. The stride of a length-1 dimension is "free", yet torch fills it with a deterministic value via rules 2/5/6 - this problem requires those filled-in values to match too.

## Examples

Each line is `op -> shape/strides/offset contig=<bool> <view|copy>`; shapes and strides are written as `(d1,d2,...)` and the base metadata heads each chain with a `base` line.

```
base (3,4)/(4,1)/0
  transpose(1,0)    -> (4,3)/(1,4)/0     contig=False view
  narrow(0,1,2)     -> (2,3)/(1,4)/1     contig=False view
  reshape(6)        -> (6)/(1)/0         contig=True  copy
  reshape(2,3)      -> (2,3)/(3,1)/0     contig=True  view

base (2,3,4)/(12,4,1)/0
  permute(2,0,1)    -> (4,2,3)/(1,12,4)/0  contig=False view
  reshape(4,6)      -> (4,6)/(1,4)/0       contig=False view
  contiguous        -> (4,6)/(6,1)/0       contig=True  copy

base (3,4)/(4,1)/0
  unsqueeze(1)      -> (3,1,4)/(4,4,1)/0      contig=True view
  unsqueeze(0)      -> (1,3,1,4)/(12,4,4,1)/0 contig=True view
  squeeze(None)     -> (3,4)/(4,1)/0          contig=True view

base (1,4)/(4,1)/0
  expand(3,4)       -> (3,4)/(0,1)/0     contig=False view
  expand(2,3,4)     -> (2,3,4)/(0,0,1)/0 contig=False view
  reshape(24)       -> (24)/(1)/0        contig=True  copy
  contiguous        -> (24)/(1)/0        contig=True  view

base (2,3,4)/(12,4,1)/0
  slice(1,0,3,2)    -> (2,2,4)/(12,8,1)/0 contig=False view
  flatten(0,1)      -> (4,4)/(4,1)/0      contig=True  copy
  flatten(0,1)      -> (16)/(1)/0         contig=True  view

base (2,3,4)/(12,4,1)/0
  select(1,-1)      -> (2,4)/(12,1)/8       contig=False view
  unsqueeze(0)      -> (1,2,4)/(24,12,1)/8  contig=False view
  narrow(1,1,1)     -> (1,1,4)/(24,12,1)/20 contig=True  view

base (4,3)/(1,4)/0
  reshape(12)       -> (12)/(1)/0      contig=True  copy
  reshape(2,6)      -> (2,6)/(6,1)/0   contig=True view
  contiguous        -> (2,6)/(6,1)/0   contig=True view

base (2,0,4)/(8,4,1)/0
  transpose(0,2)    -> (4,0,2)/(1,4,8)/0 contig=True view
  reshape(0)        -> (0)/(1)/0          contig=True view
  unsqueeze(0)      -> (1,0)/(0,1)/0      contig=True view
```

Walk-through of chain 2, step 2: for `(4,2,3)/(1,12,4)`, walking outwards from the innermost dimension, `stride[1] = 12` equals `size[2] * base = 3 * 4`, so dimensions 1 and 2 form one chunk (length 6, base stride 4), while `stride[0] = 1` differs from `6 * 4`, so dimension 0 is its own chunk. The target `(4,6)`: the 6 consumes the first chunk exactly and gets stride `1*4 = 4`, the 4 consumes the second and gets stride `1*1 = 1` - a view `(4,6)/(1,4)`. In chain 4, `reshape(24)`: the innermost chunk has length 4 (base stride 1) but the target has a single dimension of 24, so the chunk is not consumed exactly and the step is a copy; the following `contiguous` is already contiguous and therefore a view.

## Constraints

- Input size: base metadata has `ndim <= 5`, every dimension length <= 16, `stride <= 64`, `offset <= 64`, and the op sequence has at most 8 steps; the magnitude of the integers you output is unbounded.
- Everything is integer/boolean, compared exactly; no float comparison and no randomness.
- The op sequence is guaranteed executable on real torch at every step (no out-of-range indices or size conflicts); you only raise `ValueError` for inputs that violate the contract.
- Do not build tensors with torch (or numpy) and read back their `shape/stride/storage_offset/is_contiguous` to look up the answer.
