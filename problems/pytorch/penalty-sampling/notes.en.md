# Solution Notes · Repetition Penalty (Penalties)

## How the three penalties stack

llama.cpp merges the historically commonly configured `repeat_last_n / frequency_penalty / presence_penalty` into
a single `penalties` sampler (source `llama_sampler_penalties`). For each token that appeared in the window:

```cpp
if (logit <= 0) logit *= penalty_repeat;   // negative logit amplified → more negative
else            logit /= penalty_repeat;   // positive logit reduced → smaller
logit -= count * penalty_freq + (count > 0) * penalty_present;
```

- **The symmetric design of the multiplicative term** `penalty_repeat` is crucial: no matter the sign of the logit, the operation makes the logit **smaller**
  (a positive number divided by a number `>1` becomes smaller; a negative number multiplied by a number `>1` becomes more negative). This ensures the post-softmax probability necessarily decreases.
- **The freq term** grows linearly with the number of occurrences, specifically targeting "high-frequency repetition".
- **The present term** is a 0/1 step: as long as it has shown up, it gets pushed down a bit, specifically targeting "just said it, don't want to say it again".

## Window = penalty_last_n

`token_count` only counts the most recent `penalty_last_n` tokens (the source maintains a sliding window by increments and decrements: a new token
`++`, a token falling out of the window `--`, erase when it reaches zero). Passing `-1` means no window limit (the entire history). So
`win = history[-penalty_last_n:]` (when `penalty_last_n<0` use the entire history); a more robust formulation than relying on Python negative indexing is to check explicitly.

## Implementation notes

- **Order-sensitive**: the repeat multiply/divide must be done **first**, and the freq/present subtraction **afterwards**. Reversing it would let freq/present
  also be scaled by repeat, giving a different result.
- `count > 0` is always true (only tokens that appeared enter the count table), so the present term subtracts exactly once for each penalized token.
- Return only the penalized logits, **without softmax / sampling** — that is exactly its position in the pipeline (upstream of softmax).
- It is recommended to modify `logits.clone()` in place, to avoid touching the input argument.
