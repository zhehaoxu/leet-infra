# Solution Notes · H2O

## Why "cumulative" and not "average"

H2O's premise is that a token's importance is **accumulated evidence over time**, not its instantaneous attention. A token that gets a huge attention spike once and then is ignored should not be treated the same as a token that receives moderate attention at every step. Summing carries a "survival bonus" for old tokens; dividing by the number of steps would turn the score into a moving average and lose that monotone drift.

So `acc = attn.sum(dim=0)` with **no normalization whatsoever**. This is exactly where this problem differs from `kv-observed-attention`, whose notes explicitly state it divides by the number of queries. If you implement the average, the `accumulates_without_dividing` check will report `1.0` instead of `4.0`.

## Why the running sum must be per step

The score of step `s` may only use rows `0..s`. Computing the total sum once and reusing it for every step would leak the future: tokens that only become popular later would already be selected at step 0. That is why the solution keeps a `run` accumulator inside the loop and slices it `run[:P+s]`.

## Why the final column is always zero

At step `s` the newly generated token has not yet been attended to by anything—attention rows only span the positions that existed *before* the step. So column `P+s` is still zero when it is added. It can be selected later once later rows put mass on it. This is also why `P = L - S`: the last `S` columns are the tokens appended during decoding.

## Tie-breaking

`torch.argsort(..., stable=True)` on the negated score guarantees that equal scores keep their original (ascending index) order. Plain `torch.topk` gives no ordering guarantee for ties, so using it would make the output depend on the backend. The problem statement fixes "smaller index wins", and the oracle uses a Python `sort(key=lambda j: (-run[j], j))` to enforce the same rule.

## Sink tokens

Sink tokens are kept *unconditionally*, even when their cumulative score is 0. That is the whole point of the sink concept: attention sinks absorb probability mass for structural reasons (BOS, delimiters) and evicting them destabilizes the model even if their numeric score looks unremarkable.

## Complexity

\(O(S\cdot L\log L)\) for the per-step ranking, \(O(SL)\) for the accumulation. In the real system H2O maintains a heap per layer and gets amortized \(O(L + k\log L)\) per step, but the trace semantics are identical.

## Related problems

- `kv-observed-attention`: same attention statistics, but averaged rather than accumulated, and single-shot rather than per step.
- `kv-streaming-llm-sink`: keeps sink + sliding window, with no content-based score at all.
- `kv-adakv-headwise`: budget is distributed across heads instead of being a global top-k.
