# Repetition Penalty Sampling (Penalties)

## Problem

Once a model enters "broken record" mode it keeps spitting out the same words. llama.cpp's `penalties` sampler combines three
anti-repetition penalties into one step, directly **modifying the logits** (before softmax), lowering tokens that appeared recently:

- **penalty_repeat** (multiplicative, classic XPRisk / frequency penalty): for tokens that appeared in the window,
  positive logits are divided by `repeat` and negative logits multiplied by `repeat`; both effects push them **smaller**. `repeat=1.0` disables it.
- **penalty_freq** (additive): additionally subtract `count * freq` — the more occurrences, the heavier the penalty.
- **penalty_present** (additive): additionally subtract `(count>0) * present` — penalize once as long as it appeared, regardless of the count.

The three stack. Unlike the previous problems, this one **does no sampling**; it only outputs the penalized logits vector.

## Rules

- Implement `solve(logits, history, penalty_repeat, penalty_freq, penalty_present, penalty_last_n)`:
  - `logits`: 1D raw logits of shape `(V,)`.
  - `history`: list of recently generated token indices (**chronological order**, the last one is the newest).
  - `penalty_last_n`: statistics window size — only the last `penalty_last_n` tokens of `history` are considered;
    passing `-1` means the entire history is in the window.
- Steps: ① take the window `win = history if penalty_last_n < 0 else history[-penalty_last_n:]`;
  ② count the occurrences `count` of each token in `win`; ③ for each token `t` with `count>0` (`l = logits[t]`):
  $$l \leftarrow \begin{cases} l \cdot \text{repeat}, & l \le 0\\[2pt] l \,/\, \text{repeat}, & l > 0\end{cases}
  \qquad l \leftarrow l - \text{count}\cdot\text{freq} - (\text{count}>0)\cdot\text{present}$$
- Tokens that are not penalized are kept as-is. Return the **penalized** `logits` (shape `(V,)`).

## Examples

```
Input: logits = [1, -2, 0.5, 3, -0.5], history = [0, 1, 0, 2, 0],
      penalty_repeat = 1.1, penalty_freq = 0.1, penalty_present = 0.2, penalty_last_n = -1
  window counts count = {0:3, 1:1, 2:1}
  token0 (l=1.0>0): 1.0/1.1 = 0.909091; - 3*0.1 - 0.2 = -0.5 → 0.409091
  token1 (l=-2.0≤0): -2.0*1.1 = -2.2; - 1*0.1 - 0.2 = -0.3 → -2.5
  token2 (l=0.5>0): 0.5/1.1 = 0.454545; - 0.1 - 0.2 = -0.3 → 0.154545
  token3, token4 did not appear → unchanged (3.0, -0.5)
Output: penalized = [0.409091, -2.5, 0.154545, 3.0, -0.5]
```

## Constraints

- float32 input; all indices in `history` are within `[0, V)`
- The penalty order is fixed: repeat multiply/divide **first**, subtract `count*freq + present` **afterwards**
- With `penalty_repeat=1.0, penalty_freq=0.0, penalty_present=0.0` the output is always equal to the input
- Returns a 1D tensor with the same shape as `logits`
