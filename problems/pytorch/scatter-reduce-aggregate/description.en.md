# scatter_reduce Aggregated Writes (sum/prod/mean/amax/amin + include_self)

## Problem

`scatter_` overwrites and `scatter_add_` only accumulates, while genuine aggregated writing — merging several contributions into the same target slot with sum/prod/mean/amax/amin and deciding whether to fold in the original value — is the semantics of `Tensor.scatter_reduce_`. It shows up constantly in sparse aggregation, token routing (MoE unpermute) and writing variable-length segment statistics back into fixed-length tensors, and it has two behaviours that are invisible unless you read the source: mean uses an incremental (Welford-style) update rather than "total/count", and with include_self=False the touched slot is seeded with the **first scatter value**. This problem asks you to reproduce that CPU semantics point by point: given input, src, index, dim, reduce and include_self, return the written tensor. CPU, fully deterministic, no random number calls, input is not modified.

## Rules

- The complete signature, argument meanings and defaults live only in the starter docstring; that is the sole judging contract.
- **Target position**: for each coordinate $c=(i_0,\dots,i_{k-1})$ of index, the value $v$ at the same coordinate of src is written to the position pos of input whose "dim-th coordinate is replaced by `index[c]`, all other coordinates unchanged". index may contain repeated coordinates, so one pos can receive several contributions.
- **Arrival order**: contributions are folded in **index row-major order** (for a given pos, equivalently ascending by its dim coordinate), and the aggregation is a **serial** fold, not a parallel reduction — this is what makes the floating-point rounding reproducible.
- **include_self=True**: pos starts from the original input value and folds each contribution in: sum accumulates, prod multiplies, amax/amin take running max/min, and mean uses
  \[n\leftarrow n+1,\qquad m\leftarrow m+\frac{v-m}{n}\quad(\text{initial }m=\text{original input value},\ n=1),\]
  i.e. a **Welford-style incremental update**.
- **include_self=False**: pos does **not** fold in the original value but starts from the **first scatter contribution** (sum/prod accumulate from that first value; mean's initial value is that first contribution with $n=1$, then the same incremental update continues; amax/amin take extremes over scatter values only).
- **Positions never touched by any index**: always equal to the **original input value** under both include_self settings (including mean with include_self=False — it is not 0).
- The incremental mean and the batch "sum/count" formula differ in floating point, and both effects are observable:
  - **Counting**: with include_self=True the original value is both the seed and one member of the average. base=0 with contributions $[1,2,3,4]$ → **2.0**; averaging the scatter values alone gives 2.5. base=0 with contributions $[10^8,1,1,1]$ → **\(2\times10^7\)**; the same mistake gives \(2.5\times10^7\).
  - **Serial rounding**: every step updates in the input dtype, so with widely separated magnitudes the result departs from "sum first, then divide" — for base=0 with contributions $[10^8,1,1,-10^8]$ the incremental form gives **0.0** while the batch formula gives **0.4**. (The judged data has mild magnitudes and is separated mainly by the counting cases; the practical rule is to accumulate step by step in the input dtype rather than batch-computing in float64 and casting back.)
- Return a **new** tensor with the same shape and dtype as input; `solve` must not modify input in place.

## Example

input \(=[10,20,30,40]\), src \(=[1,2,3,4,5]\), index \(=[0,1,1,3,3]\) (dim=0). Contributions: pos0←1; pos1←2,3; pos2 gets nothing; pos3←4,5.

| reduce | include_self=True | include_self=False |
|---|---|---|
| sum | \([11,25,30,49]\) | \([1,5,30,9]\) |
| prod | \([10,120,30,800]\) | \([1,6,30,20]\) |
| mean | \([5.5,\,8.333333,\,30,\,16.333334]\) | \([1,\,2.5,\,30,\,4.5]\) |
| amax | \([10,20,30,40]\) | \([1,3,30,5]\) |
| amin | \([1,2,30,4]\) | \([1,2,30,4]\) |

- sum: $10{+}1$, $20{+}2{+}3$, $40{+}4{+}5$; without the self value these become $1$, $5$, $9$.
- prod: $10{\times}1$, $20{\times}2{\times}3$, $40{\times}4{\times}5$; without self, $2{\times}3{=}6$ and $4{\times}5{=}20$.
- mean (inc=True): pos1 starts at 20 → $20+\frac{2-20}{2}=11$ → $11+\frac{3-11}{3}=8.3333$; pos3 starts at 40 → $22$ → $16.3333$. mean (inc=False): pos1 is seeded with the first contribution 2 → $2+\frac{3-2}{2}=2.5$.
- **The amax pair is the contrast case**: with inc=True the original value also competes, and here every scatter value is smaller than the value already in the slot ($1<10$, $2,3<20$, $4,5<40$), so the output **equals the input**; with inc=False only scatter values compete → \([1,3,30,5]\).
- **The amin columns coincide** because in each slot the first scatter value is already below the original, so whether the self value participates changes nothing.
- The untouched pos2 is **30** in all ten cells — direct evidence that "include_self=False does not zero out" is only about touched slots.

## Constraints

- reduce ∈ {`"sum"`, `"prod"`, `"mean"`, `"amax"`, `"amin"`}; an illegal value raises **ValueError** (deliberately different from native); every other error matches native: negative index or $\ge$ input.shape[dim] → RuntimeError; dimension-count mismatch, a violated per-axis `<=` shape constraint, a floating-point index dtype, or src dtype ≠ input dtype → RuntimeError; dim outside $[-n, n)$ → IndexError
- Shape constraints: input/src/index have the **same number of dimensions**; per axis `index.shape[d] <= src.shape[d]`; for every axis except dim, `index.shape[d] <= input.shape[d]`; along dim itself index may be wider than input (repeated writes to one slot are the point of this problem)
- When src is larger than index, pairing is by **identical coordinates** using the prefix; extra src elements are ignored
- Negative dim indexing is supported; with an empty input or zero index elements, return a copy of input
- Tolerances: float32 \(10^{-5}\), float64 \(10^{-9}\) (including the incremental mean path); "untouched positions" and "fully covered single-contribution positions" are compared exactly with `torch.equal`
