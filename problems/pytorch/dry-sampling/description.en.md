# DRY Sampling (Dont Repeat Yourself)

## Problem

Ordinary repetition penalties only look at "how many times a single token has appeared" and cannot control "an entire n-gram repeating in a loop"
(`...ab ab ab ab...`). **DRY** (impl/llama.cpp's `dry`) penalizes **repeated sequences**: it finds
"the repeated substring at the end of the context" and applies an **exponentially** growing penalty to **the token that would extend the loop by one more step**
(the longer the repetition, the harsher the penalty: $\text{mult}\cdot\text{base}^{\text{repetition length}-\text{allowed}}$).

llama.cpp uses a **reverse Z-algorithm** to compute the "suffix repetition length" at every position in linear time. This problem implements its
version **without sequence breakers** (no restart-string truncation, `rep_limit = N`). As in the penalty problems, DRY **only modifies logits and does not sample**.

## Notation

- `last = history` (if `dry_penalty_last_n < 0`) or `history[-dry_penalty_last_n:]`; $N = \text{len}(last)$.
- `rev` = the reverse of `last`: $\text{rev}[j]$ is "the $j$-th token counted from the end" (equivalent to llama.cpp's `rat(j)`).
- $L_k$ (`repeat_len` at shift $k$): the **longest common prefix** length of `rev` and `rev[k:]`, i.e. the largest $n$ such that
  $\text{rev}[0\!:\!n]=\text{rev}[k\!:\!k+n]$ (you may compare one by one with a naive while loop, or use the Z-algorithm for speed; the results are identical).

## Rules

- Implement `solve(logits, history, dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n)`:
  `logits` has shape `(V,)`, and `history` is a list of token indices in chronological order.
- If `dry_multiplier == 0`, `dry_base < 1`, `dry_penalty_last_n == 0`, or $N \le$ `dry_allowed_length`:
  return the original `logits` directly (no penalty at all).
- Otherwise:
  1. For each shift $k = 1, \dots, N-1$, compute $L_k$ (as above).
  2. If $L_k \ge$ `dry_allowed_length`, the **candidate penalized token** is $\text{last}[N-k]$, with repetition length $L_k$;
     for the same token, take the **largest** $L_k$ among all hits.
  3. For each penalized token $t$ with its maximum repetition length $R$:
     $$\text{logits}[t] \leftarrow \text{logits}[t] - \text{dry\_multiplier}\cdot \text{dry\_base}^{\,\max(R-\text{allowed},\,0)}$$
     (Overflow protection: let `max_exponent = int(88.7228391 / log(dry_base))` (when `dry_base > 1.000001`);
     if the exponent exceeds it, clamp to `max_exponent`.)
- Unpenalized tokens are kept as is. Return the penalized `logits` (shape `(V,)`).

> Intuition: $\text{last}[N-k]$ is the token "one position before the start of the repeated segment"—its next occurrence is exactly what would
> extend the loop, so it is the one to penalize.

## Examples

```
Input: logits = zeros(8), history = [5,6,7,5,6,7], mult = 1.0, base = 1.5, allowed = 2, last_n = -1
  N = 6, rev = [7,6,5,7,6,5]
  L_1 = 0 (7≠6); L_2 = 1 (7=7 then 6≠7); L_3 = 3 ([7,6,5]=[7,6,5]); L_4 = 0; L_5 = 0
  only k=3 satisfies L_3=3 ≥ allowed=2 → penalized token = last[6-3] = last[3] = 5, repetition length R = 3
  exponent = R - allowed = 1, penalty = 1.0 * 1.5^1 = 1.5
Output: penalized = [0, 0, 0, 0, 0, -1.5, 0, 0]   (index 5 = -1.5)
```

```
Input: history = [0,1,0,1,0,1], everything else as above (mult=1.0 base=1.5 allowed=2 last_n=-1)
  rev = [1,0,1,0,1,0]; L_2 = 4 (prefix [1,0,1,0] matches)
  penalized token = last[6-2] = last[4] = 0, repetition length R = 4
  exponent = 4 - 2 = 2, penalty = 1.0 * 1.5^2 = 2.25
Output: penalized = [-2.25, 0, 0, 0, 0, 0, 0, 0]   (index 0 = -2.25)
```

## Constraints

- float32 input; `history` indices lie in `[0, V)`; this problem has no sequence breakers
- `logits[t] -= mult * base**(R-allowed)` is the only penalty action; different repetition lengths → penalties differ by powers of `base`
- When the same token is hit by multiple shifts, take the **largest** repetition length (highest power)
- The four cases that trigger an early exit (return the input unchanged) are listed under "Rules"
