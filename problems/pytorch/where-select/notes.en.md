# torch.where / Ternary Selection in Depth

## Definition

Element-wise conditional selection that also **supports broadcasting**:

$$ \text{out}_i=\begin{cases}x_i,& c_i\ \text{true}\\ y_i,& c_i\ \text{false}\end{cases} $$

`c` (the condition), `x`, and `y` are first unified to a common shape by broadcasting, then picked position by position. Equivalent to NumPy `np.where`.

## Direction (easy to get backwards at this level)

`torch.where(cond, x, y)`: **take x where cond is True, y where it is False**. A hand-written mask-style implementation must align the components correctly — this reference solution:

$$ \text{out}=x\cdot m + y\cdot(1-m),\quad m=\text{broadcast}(\text{cond}.\text{bool}()) $$

When $m$ is true the $x$ component is kept and $y$ is zeroed; writing it as $x(1-m)+y\,m$ reverses everything.

## Implementation and pitfalls

1. **Broadcast first, then compute**: `m, xt, yt = torch.broadcast_tensors(cond.bool(), x, y)`, otherwise `x*mask` has mismatched shapes, or the broadcasting between x/y is missed.
2. `cond` may hold arbitrary numeric values, so `.bool()` is required; convert the mask with `m.to(dtype)` before the multiply-add so it keeps the same dtype as x/y (especially important in half precision).
3. Plain `where` does not change dtype/shape, it only moves values; this differs from boolean indexing `x[cond]` (which **flattens** and loses shape) — use `where` to preserve shape, and boolean indexing to filter elements.

## Common uses

Zeroing out masked positions (attention fills masked slots with 0), numeric clipping, fixing NaN with `where(isnan, 0, x)`, and sign-based branching inside gradients. It is a vectorized ternary that is much faster than a for-else.
