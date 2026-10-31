# Explained: Layout Ledger — shape/stride/offset and the view-or-copy Decision

## Origin

On the tutorial side this is the official *View operations* guide (`docs/source/beginner/basics/views_tutorial.py`) together with the `reshape`/`view`/`expand`/`is_contiguous` notes on the `torch.Tensor` pages: they only hand you the three-bucket list (always a view / maybe a view / always a copy) and the failure condition of `view`. How the strides actually get filled, and when `reshape` silently degrades into a copy, is only in the implementation.

Implementation side (pytorch/pytorch v2.8.0, `aten/src/ATen/`), one entry per rule of this problem:

- `native/TensorShape.cpp:3884-3911` `inferUnsqueezeGeometry` → rule 2's `size[d] * stride[d]` (and 1 when appending past the last dimension).
- `native/TensorShape.cpp:3823-3860` `inferSqueezeGeometry` → rule 3: pure component deletion, the surviving strides are never recomputed.
- `native/TensorShape.cpp:1342-1370` `expand` (calling `inferExpandGeometry_dimvector`) → rule 5's right-aligned per-dimension test and the strides of the new leading dimensions.
- `native/TensorShape.cpp:1669-1695` `narrow` literally ends with `return at::slice(self, dim, start, start + length, 1)`; together with `select_symint` (`:2232`) and `slice` (`:3033-3089`) this shows rule 4 is one shared offset-advancing code path (normalisation, clamping, `end` defaulting to `INT64_MAX`, length `(len + step - 1) / step` rounded up, stride `*= step`).
- `TensorUtils.cpp:330-410` `at::detail::computeStride_impl` → rule 6's "split into contiguous subspaces, then require the target dimensions to consume each chunk exactly". This is the function behind `view()`'s `at least one dimension spans across two contiguous subspaces`.
- `native/TensorShape.cpp:1997-2018` `reshape_symint` → when `computeStride` returns `nullopt` it does `clone(Contiguous)` and then views, which is this problem's `copy`; `:4118-4150` `flatten` → the three branches (0-dim, `start == end` returning `self`, everything else delegating to reshape).
- The C-contiguity test and `contiguous()`'s "return self when already contiguous" follow `c10::TensorImpl::is_contiguous` semantics, reproduced by measurement (rule 7).

## Key points and pitfalls

1. **`view` and `reshape` share the test but differ in the verdict**: `reshape` runs `computeStride` and, when it fails, copies by itself and never raises; `view` raises instead. This problem only asks about `reshape`/`flatten`, so every step has a definite answer.
2. **The right mental model is "chunk then consume exactly", not "flatten then re-split"**: for `(3,4,2)/(4,1,12)` the chunks are `[2]` (base stride 12) and `[4,3]` (because `stride[0] = 4` equals the accumulated inner length 4 × base stride 1, those two dimensions merge into one chunk of length 12 and base stride 1). Target `(12,2)`: the 2 consumes the first chunk exactly → stride 12, the 12 consumes the second → stride 1, a view `(1,12)`. Target `(3,8)`: the 8 fits neither the 2 nor the 12 → copy. Target `(24,)`: spanning two chunks is not allowed either → copy. The "multiply the sizes then split" intuition gets all three wrong.
3. **The transpose/expand cases that look one step apart**: for `(4,3)/(1,4)` (a transposed matrix) both `reshape(12)` and `reshape(2,6)` are **copy** (row-major traversal really reads 0,4,8,1,5,9…, no single stride covers it, and the two 6s do not consume any chunk), while `reshape(2,2,3)` - which only splits the outer dimension - is a view `(2,1,4)`. A stride-0 dimension from `expand` always forces a copy the same way (`(3,4)/(0,1)` with `reshape(12)` or `reshape(1,12)`), whereas `(4,2,3)/(1,12,4)` with `reshape(4,6)` does not copy. Example chains 1, 2, 4 and 7 are exactly these pairs - and in chain 7 the `reshape(2,6)` right after the copying `reshape(12)` is a view again, because the fresh storage is standard row-major.
4. **The stride of a length-1 dimension is "free", but torch fills it deterministically**: `unsqueeze` takes the span of the segment to its right, a new leading 1 from `expand` takes `size[i+1] * stride[i+1]` (and therefore 0 if its neighbour is a stride-0 dimension), and the standard contiguous strides after a copy do not advance the accumulator over dimensions of length ≤ 1. A problem that compares shapes but not strides silently misses this whole layer.
5. **`squeeze` deletes components and never recomputes**: `(2,1,3)/(9,9,1)` dropping dimension 1 gives `(2,3)/(9,1)` (not `(3,1)`), so `is_contiguous()` is false; naming a dimension whose length is not 1 makes the whole step a no-op - `squeeze(0)` on `(2,1,3)/(3,3,1)` returns it unchanged. In example chain 3 the two `unsqueeze`s followed by `squeeze(None)` remove exactly the dimensions that were inserted and land back on `(3,4)/(4,1)`.
6. **An empty slice still advances the offset and still multiplies the stride by step**: `slice(1, 5, 1, 2)` on `(6,8)/(8,1)` gives `(6,0)/(8,2)` with offset 5 and `is_contiguous() == True` (zero total length is always contiguous). Same for `narrow(0, 6, 0)`, whose offset becomes 48.
7. **The zero-numel branch**: `computeStride_impl` never returns `nullopt` when `numel == 0`, so `reshape` is always a view there - strides copied from the old shape if it is identical, otherwise the standard contiguous strides (`(2,0,4)/(8,4,1)` with `reshape(3,-1)` gives `(3,0)/(1,1)`). `reshape` on a 0-dim base takes the `oldshape.empty()` branch and returns all-1 strides.
8. **`contiguous()` is not "always a copy"**: when the layout is already contiguous it returns itself and the offset survives (`(2,3)/(3,1)/4` stays at 4); only a non-contiguous layout allocates fresh storage and resets the offset to 0. When the verdict is `copy`, both things happen at once - offset 0 and standard contiguous strides.
9. **`flatten(a, a)` returns `self` unchanged**: it never reaches `computeStride`, so even a dimension sitting on a chunk boundary does not force a copy; a 0-dim tensor first goes through `reshape({1})`. Example chain 5 runs `flatten(0,1)` twice: the first call merges the leading two dimensions of `(2,2,4)/(12,8,1)`, and `stride[0] = 12 ≠ 2 × 8` proves they are in different chunks → copy to `(4,4)/(4,1)`; the second call gets an already contiguous two-dimensional input → view.

## Real behaviour outside the defined domain (deliberately excluded)

- The op set has no `view`/`_reshape_alias`/`as_strided`/`movedim`/`unfold`/`diagonal`: the first two share `reshape`'s test but raise, so they are not total as problem operations; `as_strided` writes metadata directly, which would put the answer in the input; the last three are windowed geometric transforms well beyond a layout ledger.
- `Tensor.slice` does not exist on the Python side (measured `AttributeError`); torch slicing goes through basic indexing `t[(slice(None),)*d + (slice(s,e,st),)]`, which is what the test oracle uses. The problem collapses that into one `["slice", d, start, end, step]` with `end` always explicit, so the omitted-`end` (`INT64_MAX`) branch is not examined.
- `memory_format` (`channels_last` and friends) and the single-dimension `torch.Tensor.stride(dim)` query are out of scope; only C-contiguity is discussed here.
- Strides on zero-length dimensions really can differ between backends/versions - the comment inside `computeStride_impl` says "stride is arbitrary in the numel() == 0 case". This problem pins them to the same formula torch 2.14.1 uses, i.e. implementation-as-spec.

## Cross-check methodology

The solution is pure-Python metadata arithmetic and calls no torch API. `test_cases.py` uses torch as the oracle:

- Base metadata is rebuilt with `torch.arange(n).as_strided(shape, strides, offset)` where `n = offset + Σ(size-1)*stride + 1` (clamped with `max(n, 1)`), so any metadata - overlapping, stride-0, whatever - becomes a real tensor as long as it stays inside the storage.
- `kind` is decided by whether `untyped_storage().data_ptr()` changes. This cannot give a false negative within this op set: with `numel == 0` `computeStride` never fails, and `contiguous()` returns self when already contiguous, so neither produces new storage.
- Random chains: 7 fixed seeds × 120 chains, base ndim ≤ 4 and length ≤ 8, i.e. the same domain as the constraints, about 3800 steps, each step compared on all five fields `shape/strides/offset/contiguous/kind`; the randomness comes from `random.Random(seed)` with no global state touched. The generator only emits steps torch accepts (it retries another candidate otherwise), so no exception text has to be reproduced.
- The 8 example chains come from two-way measurement against both the reference implementation and torch, and the 28 dedicated assertion chains are additionally replayed through torch.

## Complexity

O(ndim) per step, O(chain length × ndim) per call; the full cross-check finishes in under a second locally.
