# Approach

## Interviewer's Intent

"All-Reduce = Reduce-Scatter + All-Gather" is the most frequently quoted sentence in distributed communication. To understand it, you must first fully grasp each of the two halves. The output of Reduce-Scatter **differs on every rank** (each gets one segment of the reduction result), which is exactly the opposite of All-Gather's "everyone identical" — interviewers love to set traps here.

## Two-step Breakdown

1. **Reduce**: reduce (sum) $W$ copies of equal-length data element-wise into a single result $T$ of length $L$.
2. **Scatter**: cut $T$ into $W$ segments, sending the `i`-th segment to rank `i`.

So the output of rank `i` is $=T[i\cdot\text{chunk}:(i+1)\cdot\text{chunk}]$, with `chunk = L/W`. What is returned is a list where **each rank's result differs**.

## Relationship to Ring All-Reduce

Ring All-Reduce (see the communication-volume analysis in `parallel-allreduce`) is exactly:
- Phase one, Reduce-Scatter: after $W-1$ laps, each GPU holds exactly $\frac1W$ of the "global sum";
- Phase two, All-Gather: another $W-1$ laps to stitch each GPU's $\frac1W$ together for everyone → each GPU obtains the complete global sum.

Each step's communication volume is $(W-1)/W\cdot S$, and together they give $2\frac{W-1}{W}S$.

## Real-world Usage: ZeRO

The communication of ZeRO-1/2 is exactly Reduce-Scatter: optimizer states/gradients are cut into $W$ pieces, each GPU is responsible only for updating its own $\frac1W$, which naturally amortizes optimizer memory down to $1/W$ (corresponding to the partitioning formula in `training-state-memory`).

## Common Mistakes

- Returning the complete $T$ on every rank (that is All-Reduce, not Reduce-Scatter).
- Wrong slicing order (giving the `i`-th segment to rank `W-1-i`).
- Using `reshape` without ensuring divisibility by $W$ (this contract guarantees divisibility, but you should still compute chunk explicitly).

## Grading Approach

`test_cases.py` checks against an independent reference that sums and slices element-wise by hand, and pins down the invariants: **concatenating the per-rank shards == the complete element-wise sum**, correct rank order, and no modification of the inputs.
