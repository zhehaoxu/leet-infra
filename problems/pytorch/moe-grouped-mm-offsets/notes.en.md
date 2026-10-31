# Reference Approach: One Host Sync for offs, Loop over Groups, Three Sliced Matmuls Each

## Implementation

1. Validate the way `_grouped_mm_fallback_fake` does, then add the three conventions that belong to this statement: `offs` is non-decreasing, its entries are non-negative, and its last entry does not exceed the row count. Do all shape checking *before* the first `torch.mm`; otherwise "the gradient direction is transposed whenever K differs from N" leaks out as a `RuntimeError` instead of a clean `ValueError`.
2. Allocate `out`, `grad_input` and `grad_weight` with `torch.zeros` and then **overwrite only the rows that have an owner**. The tail rows in $[\mathit{offs}[E-1],S)$ and the `grad_weight[e]` of empty groups end up zero by construction — do not special-case them.
3. Call `offs.tolist()` once, outside the loop, and use host-side integers inside it. `counts` comes from differencing: $\mathit{count}_e=\mathit{offs}[e]-\mathit{offs}[e-1]$, with the left edge taken as 0 for $e=0$.
4. Per group, three `torch.mm` calls, each writing into a preallocated slice with `out=`: forward `x @ w`, input gradient `g @ w.T`, weight gradient `x.T @ g`. The **left operand differs** between the last two, which is the single easiest thing to write backwards.
5. Skip empty groups with `if start == end: continue`, keeping `start = end` on the way through. Both statements are no-ops here; they stay to show the reader that the partition did not stall.

Four common mistakes:

- **Reading `offs` as group sizes** (`end = start + offs[e]`) or **as group starts** (`table = offs + [S]`, group $e$ = `table[e:e+2]`). The first is correct only for group 0; the second slides the whole dispatch one slot to the left and, at the far end, silently swallows the tail rows.
- **Writing `grad_weight` as `g.T @ x`**: that yields $[N,K]$. When $K\neq N$ the shape check stops it; when $K=N$ the shape is perfectly legal and only the values are transposed, so the judge contains both kinds of case.
- **Starting from `torch.empty`**: see the mutation-testing result below — the judge catches this unreliably.
- **Forgetting that `weight[e]` is indexed by group id**: implementations that renumber only the non-empty groups drift as soon as one sits in the middle. The judge pins this with "swap an empty expert's weight for random values".

## Correctness checks

Eight `check_*` functions: the hand-computed example (all four return values verified, empty group and tail rows included), random differential testing (8 combinations of shape/assignment/magnitude, one of them with genuine `float64` random data, plus `offs` as int32 and three non-contiguous views built for `x`, `weight` and `grad_output` separately), group algebra invariants (permuting rows inside a group must permute `out`/`grad_input` the same way while `grad_weight`/`counts` stay bit-identical; **replacing an empty expert's weight with random values must leave all four return values unchanged**; `counts` agrees with the differenced `offs` and sums exactly to $\mathit{offs}[E-1]$), all-empty and all-taken (`offs` all zeros, only the first group non-empty, only the last, two empty in the middle, single expert, int32), tail rows (zero for three different `offs`, plus four values checked explicitly on a small hand case), exactness (quarter-multiple data compared bit-for-bit with `rtol=atol=0` against the pure-Python scalar triple loop), invalid inputs, and the maximum shape of 32 rows over 16 experts.

The oracle is pure Python: a scalar triple loop over groups, rows and columns that calls no `torch` matmul and does not reuse the slicing formulation of the reference solution. Integers and quarter multiples are binary-exact, so the bit-for-bit claim is not smuggled through a tolerance.

Mutation testing ran 16 mutants and caught 14: `offs` read as sizes, `offs` read as starts, the last group swallowing the tail, `counts` set to `offs` itself, `counts` off by one (counting the right endpoint), input gradient without the transpose, input gradient built from `x` instead of `g`, weight gradient as `x @ g.T` (still $[K,N]$, values completely misplaced), weight gradient as `g.T @ x`, weight gradient computed correctly then transposed, backward slicing replaced by the prefix $[0,\mathit{end})$, every group sharing `weight[0]`, singleton groups skipped, and `torch.empty` allocation. Two **cannot be distinguished**:

- **An empty group that "multiplies anyway" using the next expert's weight** — `x[start:start]` is 0 rows, `torch.mm` writes 0 rows, and no weight takes part in the arithmetic. This is a **genuine equivalence**, and the statement now says so outright: what the judge catches is a wrong `weight` index on a **non-empty** group (the `weight_shared` mutant is exactly that, and it is caught), not whether an empty group multiplies.
- **`torch.empty` allocation** — of the 16 runs only `check_maximum_shape` failed, because small allocations on this machine happen to land on zero pages. I added "fill a same-shaped buffer with non-zero data and drop it first" to both `check_all_empty_and_full_offsets` and `check_tail_rows`, and it still does not surface reliably. The honest conclusion: **this convention is not guaranteed by the judge**. It is fixed by the statement and must be upheld by the implementation; the judge only errors when the memory layout happens to be dirty.

## Scope and sources

The partition convention, skipping empty groups, zero tail rows and the three forward/backward matmuls are all taken from `integrations/moe.py` in the transformers 5.20.2 installed for this repository's judge — that code is the fallback path actually executed when grouped-mm is unavailable (non-bfloat16 weights, `torch.compile` fullgraph, no matching kernel), and I lifted the assertion list of `_grouped_mm_fallback_fake` into the validation nearly verbatim (adding only monotonicity and the range check on `offs`). Verbatim excerpt:

```text
# transformers/integrations/moe.py
def _grouped_mm_fallback(input, weight, offs):
    output = torch.zeros(input.size(0), weight.size(2), ...)
    start = 0
    # single cpu<->gpu sync point here,
    # avoids multiple syncs inside the loop
    for i, end in enumerate(offs.tolist()):
        if start == end:
            continue
        torch.mm(input[start:end], weight[i], out=output[start:end])
        start = end
    return output

def _grouped_mm_fallback_fake(input, weight, offs):
    assert input.dim() == 2
    assert weight.dim() == 3
    assert offs.dim() == 1
    assert offs.size(0) == weight.size(0)
    assert input.size(1) == weight.size(1)
    assert offs.dtype in (torch.int32, torch.int64)

def _grouped_mm_fallback_backward(ctx, grad_output):
    input, weight = ctx.saved_tensors
    grad_input = torch.zeros_like(input)
    grad_weight = torch.zeros_like(weight)
    start = 0
    for i, end in enumerate(ctx.offs.tolist()):
        if start == end:
            continue
        torch.mm(grad_output[start:end], weight[i].T, out=grad_input[start:end])
        torch.mm(input[start:end].T, grad_output[start:end], out=grad_weight[i])
        start = end
    return grad_input, grad_weight, None
```

Two things are **defined by the problem statement**, with no attribution claimed: the `counts` return value and its int64 typing (the upstream function has no such output), and turning invalid input into `ValueError` (upstream uses bare `assert`). Upstream hangs these three pieces on the forward/backward of a `torch.library.custom_op` and lets autograd drive them; this problem flattens that into a pure function — no computation graph, no `backward()` call, gradients treated as ordinary matmuls.

Division of labour inside the collection: `moe-grouped-execution-plan` decides **how to arrange** these groups (packing ragged groups into fixed-size tiles, padding and scheduling cost); this problem covers **how to compute once they are arranged**. Getting tokens into the row order that `x` assumes is the job of `moe-all-to-all-dispatch` together with the two capacity problems (`moe-capacity-dispatch`, `moe-capacity-jitter-dispatch`); the semantics of the expert weights themselves (shared expert, up/down projections) live in `moe-shared-expert-forward`.
