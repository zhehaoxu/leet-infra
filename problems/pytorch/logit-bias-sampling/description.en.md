# Sampling: Logit Bias and Banned Tokens

## Problem

The two most common out-of-band instructions when tuning sampling are: **nudge** a token's likelihood up or down, and **forbid** a token from ever appearing. The first is OpenAI's `logit_bias` — a "token → bias value" map; the second is every engine's banned / suppress list. They look alike ("touching one token's score") but are two different kinds of mathematics: a bias is a **translation** of a logit **before** softmax, changing only the gap between that entry and the others while leaving the relative structure of all other tokens untouched; a ban **deletes an element from the support set**, and only \(-\infty\) makes its probability **exactly** \(0\).

This problem composes the two steps into one operator: bias map → hard mask → softmax → a single sample. It is also the very front of llama.cpp's sampling chain (user biases and suppress tokens are merged into one sampler, executed before any truncation or temperature), but **only the mathematics of this one step is examined here**: no temperature, no top-\(k\)/top-\(p\), so there is no ordering ambiguity.

## Rules

- Inputs: `logits` (shape \((V,)\) or \((\dots,V)\), the last dimension being the vocabulary), one bias map (two equal-length lists: token indices and the matching bias values — entry \(i\) adds `bias_value[i]` to index `bias_token[i]`), one list of banned indices `banned`, and a uniform draw \(u\) (scalar, single-element, or one per row, valued in \([0,1)\)). **Randomness may only come from \(u\).**
- **① Additive bias, applied entry by entry in list order**: \(\ell_t \leftarrow \ell_t + \delta\). If the same index appears several times the contributions **accumulate one by one** (no merging: `+0.5` then `+1.5` equals `+2.0`). **Entries whose index is out of range or negative have no side effect at all** — no error, no clamping, no modulo, and they do not disturb the other entries of the map.
- **② Hard mask**: set the logit at every index of `banned` that falls in \([0,V)\) to **\(-\infty\)**; out-of-range/negative indices are skipped here too. This step runs **after** ①, so an index hit by both lists ends up at \(-\infty\) (the mask overrides the bias).
- **③ softmax**: apply softmax to the logits after biasing and masking. Banned entries get probability **exactly \(0\)**; the rest are normalized over the **surviving set** (equivalent to "delete those tokens from the vocabulary first, then softmax", not "softmax over the whole vocabulary and then fix up the normalization").
- **④ A single sample**: take the inverse CDF of `filtered` in the **original index order** with \(u\) (the first index whose cumulative sum is \(\ge u\)) to get `chosen`; `chosen` necessarily lands on a non-banned position.
- Outputs, three of them: `biased_logits` (same shape as `logits`, contains \(-\infty\), and is a new tensor rather than an in-place edit of the input), `filtered` (each row sums to \(1\), banned positions are \(0\)), `chosen` (one per row; a 0-dimensional tensor for a single row).

## Example

```
Input: logits = [1.0, 0.5, 0.0, -0.5, -1.0]
       bias map: indices [1, 1, 3] with values [0.5, 1.5, -2.0] (applied one by one, in list order)
       banned: index [4]
       u = 0.6

① bias: index 1 appears twice → 0.5 + 0.5 + 1.5 = 2.5; index 3 → -0.5 - 2.0 = -2.5; indices 0 and 2 are not in the map → unchanged
② mask: index 4 → -inf
   biased_logits = [1.0, 2.5, 0.0, -2.5, -inf]
③ softmax (normalized over the surviving set; the banned entry is exactly 0)
   filtered = [0.170075, 0.762222, 0.062567, 0.005136, 0.0]
   (with no bias and no ban it would be [0.428656, 0.259993, 0.157694, 0.095646, 0.058012])
④ CDF of filtered in original order = [0.170075, 0.932297, 0.994864, 1.0, 1.0], u = 0.6 → first ≥ 0.6 is index 1

Output: biased_logits = [1.0, 2.5, 0.0, -2.5, -inf]
        filtered = [0.170075, 0.762222, 0.062567, 0.005136, 0.0]
        chosen = 1
```

Batched and boundary cases:

```
Batched (both rows share the same bias map and banned list, one u per row):
  logits = [[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]], bias map indices [0, 2] / values [-3.0, 1.0], banned [1], u = [0.2, 0.9]
  biased_logits = [[-1.0, -inf, 1.0], [-3.0, -inf, 3.0]]
  filtered = [[0.119203, 0.0, 0.880797], [0.002473, 0.0, 0.997527]]
  per-row CDF in original order = [[0.119203, 0.119203, 1.0], [0.002473, 0.002473, 1.0]] → chosen = [2, 2]

Entries with out-of-range / negative indices are complete no-ops:
  logits = [1.0, 0.0, -1.0], bias map indices [-1, 7, 999] / values [5.0, -5.0, 100.0], banned [-3, 42], u = 0.2
  → result is identical to an "empty map": biased_logits = [1.0, 0.0, -1.0],
    filtered = [0.665241, 0.244728, 0.090031], chosen = 0

Banning only the largest entry renormalizes the survivors:
  logits = [5.0, 4.0, 3.0, 2.0], empty bias map, banned = [0], u = 0.5
  → filtered = [0.0, 0.665241, 0.244728, 0.090031] (index 0 is exactly 0; chosen can never be 0)
```

## Constraints

- Judging environment: **CPU only**; **torch / numpy only**; **completely offline** — no llama.cpp, no transformers sampler / logits-processor classes, no network dependency of any kind. **All randomness must be an explicit argument**: the uniform draw arrives as \(u\); you may not call `manual_seed`, `torch.rand` or `torch.multinomial`.
- \(V \ge 1\); the two lists of the bias map have equal length (the map may be empty); indices are integers, and anything outside \([0,V)\) is a no-op.
- **Each row must keep at least one non-banned token**: a row that is entirely \(-\infty\) makes "subtract the maximum" evaluate to \(\infty-\infty=\text{NaN}\), and that case is outside the judging scope.
- The execution order is fixed as **① bias → ② mask → ③ softmax → ④ sample**. The mask is an **assignment**, not an addition, so when both lists hit the same index, \(-\infty\) wins. Never move the bias to **after** softmax as an addition/subtraction on probabilities — that is no longer a logit bias (see the convention comparison in `notes.md`).
- Numerics: float32 intermediates, judged with `atol = rtol = 1e-5`; each row of `filtered` sums to \(1\) and banned positions are **exactly** \(0.0\).
