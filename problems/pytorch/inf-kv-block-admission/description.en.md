# KV Block Budget Admission Control with Stalled Decodes

## Problem

Every round of a continuous-batching engine answers two questions: can the requests **already running** emit one more token, and can the requests **still waiting** be pulled into the running set. Both answers come from the same resource: how many KV blocks are left.

A request needs `ceil(prompt_len / block_size)` blocks for prefill; afterwards each generated token needs a new block only when it crosses a block boundary. When blocks run short, production engines preempt (swap or recompute — see the neighbours of `inf-lru-kv-cache-eviction`). This problem assumes **no preemption**: a decode that cannot get its block merely stalls for this step and retries next step.

The other easy-to-miss point: admission is *not* "stop at the first request that does not fit" — one large prompt would block the whole queue. Scan the whole waiting list, start everything that fits, skip the rest and keep looking.

## Task

Implement `solve(num_blocks, block_size, prompt_lens, max_new, max_steps)`:

- Inputs: `num_blocks` physical blocks; `block_size` tokens per block; `prompt_lens` `(N,)` int64; `max_new` `(N,)` int64; `max_steps` simulation limit.
- Each step runs decode first, then admission; both phases walk requests in ascending id:
  - **Decode**: for an admitted, unfinished request, the blocks needed after emitting one token is `ceil((prompt_len + generated + 1) / block_size)`. Take the extra blocks from the free pool; if they are not there, stall this step and emit nothing.
  - **Admission**: for a waiting request, prefill needs `ceil(prompt_len / block_size)` blocks. Admit and record the step if it fits; otherwise skip.
- When a request finishes (`generated == max_new`) it returns all its blocks immediately.
- Outputs: `admit_step` `(N,)` (-1 = never admitted), `finish_step` `(N,)` (-1 = unfinished), `stalls` `(N,)` (stalled steps per request), `peak_used` int64 scalar (blocks held at the tightest moment).

## Example

```
num_blocks = 3, block_size = 4
prompt_lens = [4, 6], max_new = [4, 2]

step 0  admit: r0 needs 1 block -> admitted (holds 1); r1 needs 2 -> admitted (holds 2)  used = 3
step 1  r0 wants token 1 -> needs ceil(5/4)=2 blocks, 0 free -> stall
        r1 wants token 1 -> needs 2 blocks, already holds 2 -> emits
step 2  r0 stalls again (2 total)
        r1 emits token 2 -> done, returns 2 blocks
step 3  r0 gets 1 block (holds 2), emits token 1
step 4..6  r0 emits the remaining tokens, finishes at step 6

admit_step  = [0, 0]
finish_step = [6, 2]
stalls      = [2, 0]
peak_used   = 3
```

## Constraints

- **Phase order is fixed**: decode then admission. Blocks returned by a request that finished this round are immediately usable by a waiting request in the same round.
- **Ascending id inside each phase**: whoever grabs a block first decides who gets one later, so it must be deterministic.
- **No preemption**: a blocked decode stalls; it never evicts anyone (that is a different policy).
- **Admission must not head-of-line block**: skip what does not fit and keep scanning.
- **`max_new = 0`**: admission is completion, so `finish_step == admit_step`.
- **Requests still unfinished when `max_steps` runs out**: `finish_step` is `-1`, and the stalls they already accumulated are still reported.
