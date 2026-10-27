# Solution Notes · Block Budget Admission and Stalling

## Fix the skeleton first, then the details

Ninety percent of the correctness here is obeying the skeleton: **inside one step, decode first and admit second, and both sub-loops walk ids ascending**. Swap either and the outputs drift:

- Admit before decode and blocks returned by a request that just finished are only reusable next round, pushing every `admit_step` later.
- Walk the decode phase backwards and the blocks go to later requests first, so which request gets charged the stall changes.

## Blocks held is a function of the token count, not a running total

Do not keep a "blocks used" counter and `+1` it. Recompute from the token count every step:

```
need = ceil((prompt_len + generated + 1) / block_size)
extra = need - held
```

Blocks are per **page**: when `prompt_len` is not a multiple of `block_size`, the first few generated tokens still land in the same page and `extra` is 0. Only crossing a boundary asks for a real block.

## Finishing returns the blocks

The moment `generated == max_new`, add `held` back to the free pool and zero it. Miss this and later requests starve forever, turning their `finish_step` into `-1`.

## Record the peak at allocation time

The peak is not the final occupancy but the maximum over all moments. The robust way is to update `peak = max(peak, num_blocks - free)` right after every successful allocation, not at the end of a phase, where returned blocks may already have restored free space.

## The two classic bugs

1. **`break` on the first waiting request that does not fit**: this degenerates into head-of-line blocking, and one huge prompt starves every small request behind it. Use `continue`.
2. **Treating `max_new == 0` as invalid**: it is meaningful in an engine (prefill only, drop right after the first token), and it must finish in its admission step and return its blocks.

## Complexity

Two O(N) scans per step, so O(max_steps × N) overall. Production engines keep deques plus block counters, but that is a constant-factor optimisation and does not change the judging contract.

## Related problems

- `inf-continuous-batching-scheduler`: same running set, but counts against a fixed `max_batch_size` and models no block budget.
- `inf-paged-kv-block-alloc`: where blocks come from and how they are handed out in order.
- `inf-paged-cow-block-share`: bookkeeping once several sequences share blocks.
