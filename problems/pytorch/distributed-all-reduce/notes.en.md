# Approach

## Interviewer's Intent

All-Reduce is the lifeblood of data parallelism: each GPU computes the gradients for its own mini-batch, and then must **sum the gradients and synchronize them to all GPUs** so that everyone updates the same model. "What exactly does DDP communicate in one step, and why is it All-Reduce" is a must-ask question for training roles.

## Semantics: Reduction + Complete Result Everywhere

The only difference between All-Reduce and Reduce-Scatter is the **output volume**:
- Reduce-Scatter: each rank gets only $\frac1W$ of the reduction result;
- All-Reduce: each rank gets the **complete** reduction result (the All-Gather step is also performed).

$$ T = \sum_{r} \texttt{inputs[}r\texttt{]},\quad \text{every rank receives } T $$

## Identity: All-Reduce = All-Gather ∘ Reduce-Scatter

The test `check_equals_gather_of_scatter` pins this relationship down: first reduce-scatter so that each GPU holds one segment of the global sum, then all-gather to stitch all segments back together, yielding the complete sum on every GPU — exactly the two phases of Ring All-Reduce. Once you understand this identity, the $2\frac{W-1}{W}S$ communication volume in `parallel-allreduce` follows directly from "one RS lap + one AG lap".

## Gradient Averaging vs Summation

DDP actually uses **gradient averaging**: after NCCL's All-Reduce sums, the framework divides by `world_size`. This problem follows the standard semantics of the collective primitive and only performs **summation**; to average, simply divide $T$ by $W$ uniformly (pure post-processing that does not change the communication structure).

## Common Mistakes

- Computing the sum but returning only one tensor (you should return a list of length $W$, each item being $T$).
- Writing averaging instead of summation (this contract is summation).
- Using `+=` to modify `inputs[0]` in place, polluting the input.

## Grading Approach

`test_cases.py` checks against an independent reference that accumulates element-wise in Python, pinning down: all ranks are identical and hold the complete sum, the composition identity **All-Reduce == gather(scatter(...))**, exact integer arithmetic, single-rank degeneration, and no modification of the inputs.
