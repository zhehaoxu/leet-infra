# Solution Notes · H2O

## Why "cumulative" rather than "average"

H2O's premise is that a token's importance is **accumulated evidence over time**, not its instantaneous attention. A token that receives one huge spike and is then ignored should not be treated like a token that receives moderate attention at every step. Summing carries a survival bonus for old tokens; dividing by the number of steps would turn the score into a moving average and destroy that monotone drift.

So `acc = attn.sum(dim=0)` with **no normalization whatsoever**. This is precisely where this problem differs from `kv-observed-attention`, whose notes state explicitly that it divides by the number of queries. An averaged implementation makes the `accumulates_without_dividing` check report `1.0` instead of `4.0`.

## Why the running sum must be per step

The score at step `s` may only use rows `0..s`. Computing the total sum once and reusing it for every step would leak the future: tokens that only become popular later would already be selected at step 0. The reference solution therefore maintains a `run` accumulator inside the loop and slices it as `run[:P+s]`.

## Why the last column is always zero

At step `s` the newly generated token has not been attended to by anything yet—an attention row only spans the positions that existed *before* the step. Hence column `P+s` is still zero when it is appended. It can be selected later once subsequent rows put mass on it. This is also why `P = L - S`: the last `S` columns are the tokens appended during decoding.

## Tie-breaking

`torch.argsort(..., stable=True)` applied to the negated score keeps equal scores in ascending index order. `torch.topk` gives no ordering guarantee for ties, so relying on it would make the output backend-dependent. The problem statement fixes "smaller index wins", and the oracle enforces it with a Python `sort(key=lambda j: (-run[j], j))`.

## Sink tokens

Sink tokens are kept *unconditionally*, even when their cumulative score is 0. That is the point of the sink concept: attention sinks absorb probability mass for structural reasons (BOS, delimiters), and evicting them destabilizes the model even when their numeric score looks unremarkable.

## Complexity

\(O(S\cdot L\log L)\) for the per-step ranking and \(O(SL)\) for the accumulation. The real system keeps a heap per layer for an amortized \(O(L + k\log L)\) per step, but the trace semantics are identical.

## Related problems

- `kv-observed-attention`: the same attention statistics, but averaged instead of accumulated, and one-shot instead of per step.
- `kv-streaming-llm-sink`: keeps sink plus a sliding window with no content-based score at all.
- `kv-adakv-headwise`: the budget is split across heads instead of being a single global top-k.
