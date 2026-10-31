# KV Cache: SCOPE's Decoding-Window Budget and Discontinuous Eviction

## Problem

SCOPE (arXiv 2412.13649, ACL 2025) splits KV compression in two: the prefill pass uses SnapKV's clustered top-k to pick a prompt cache, then the **decoding phase** applies a window schedule that grows with the step count. Upstream `model/kv_utils.py::SnapKVCluster.update_kv_in_decoding` branches on four `decoding_metric` values, and two of those branches drive the budget from **class-level counters** — the counters advance on *each layer call*, not on each token, so one decoded token pushes them by `num_hidden_layers`.

The prefill side (`pooling` / `topk` / cluster folding) is line-for-line the same as `kv-snapkv-pooling`, so **this problem only asks about the decoding schedule**: given the attention scores of each decoding step, how does the window budget grow, which steps actually evict, and which positions remain in each head's cache at the end.

Let `dw` be this step's decoding window budget, `rec` the always-kept recent window, and `mcp` the frozen prefill prefix length:

- `None`: returns immediately, no compression, `dw` recorded as 0.
- `slide`: `dw = decoding_window_size`, constant; once the cache is full it re-selects.
- `adaptive`: `dw = rec + current_decoding_step // (delta * num_hidden_layers)`, then `current_decoding_step += 1` (once per layer call), so the budget only grows by 1 every `delta` tokens.
- `discontinuous`: same budget formula as `adaptive`, but a second burst gate (`jump_step` / `jump_layer`) is stacked *after* the budget gate. `jump_step` advances by 1 per layer call and eviction only happens once it reaches `delta * num_hidden_layers`; when eviction starts, each layer evicts once and `jump_step` resets only after `num_hidden_layers` such calls. At token granularity: `delta` tokens accumulate, then the `delta + 1`-th token evicts once in every head.

## Task

Implement `solve(init_positions, attn, decoding_metric="discontinuous", num_hidden_layers=32, delta=15, max_capacity_prompt=512, decoding_recent_size=128, decoding_window_size=1024)`:

- `init_positions`: int64 1-D tensor of length `P >= 1` — the positions held by each cache slot **after** prefill compression. Values are distinct and lie in `[0, P + T)`; the order is the slot order and is **not required to be ascending**.
- `attn`: float 3-D tensor `[T, H, P + T]`, where `attn[t, h, p]` is the attention weight step `t`'s query in head `h` gives position `p`; only the ordering matters. The `t`-th newly generated token has position `P + t`.
- `decoding_metric`: one of `"None"` / `"slide"` / `"adaptive"` / `"discontinuous"`, case-sensitive.
- The remaining five are integers `>= 1`, with `max_capacity_prompt <= P`.

Each decoding step `t` (over `T` steps; `L` = cache length after appending):

1. Append position `P + t` to the tail of **every** head's cache.
2. Set the budget: `slide` uses `decoding_window_size`; `adaptive` / `discontinuous` use `rec + cds // (delta * num_hidden_layers)` and then add `num_hidden_layers` to `cds` (starting from 0 — equivalent to upstream's "+1 per layer call"); `None` skips the schedule, records budget 0 and never evicts.
3. Gate: eviction is only possible when `L >= max_capacity_prompt + dw`. `discontinuous` stacks the burst gate — when `jump >= delta * num_hidden_layers`, reset `jump` to 0 and allow eviction; otherwise add `num_hidden_layers` to `jump` and skip eviction for this step. `slide` / `adaptive` have no such gate.
4. Evict: `keep = dw - rec`; the candidate region is `slots[max_capacity_prompt : L - rec]` (the frozen prefix and the trailing `rec` slots are kept unconditionally and never compete); take the `keep` entries with the highest `attn[t][h][position]` in **descending** order, breaking ties by the earlier slot in region order; the new cache = frozen prefix + the chosen entries (left in descending score order, not re-sorted by position) + the trailing window. Each head selects independently using its own score row.
5. Record the cache length at the end of the step and whether the step actually evicted.

Return a 4-key dict of int64 tensors: `window_sizes` shape `[T]` (each step's `dw`), `cache_lens` shape `[T]` (length at end of each step; all heads are equal-length, so head 0 suffices), `evict_steps` shape `[E]` (step numbers that evicted, ascending), `kept_positions` shape `[H, final_length]` (positions left in each head's cache, in slot order).

Raise `ValueError` when `decoding_metric` is not one of the four values; when any of the five integers is not an `int` (`bool` counts as illegal) or `< 1`; when `max_capacity_prompt > P`; when `attn` is not a 3-D float tensor, or its width differs from `P + T`; when `T < 1`; when `init_positions` is not 1-D, is empty, has duplicates or out-of-range values; and when `decoding_metric == "slide"` with `decoding_window_size <= decoding_recent_size`.

## Example

```
P = 6, T = 6, H = 1, num_hidden_layers = 2, delta = 2
max_capacity_prompt = 4, decoding_recent_size = 3, decoding_window_size = 4
init_positions = [0, 1, 2, 3, 4, 5]
attn[t][0], columns = positions 0..11
  t=0   0.05 0.06 0.07 0.08 0.09 0.10 0.11 0.12 0.13 0.14 0.15 0.16
  t=1   0.16 0.05 0.06 0.07 0.08 0.09 0.10 0.11 0.12 0.13 0.14 0.15
  t=2   0.10 0.11 0.12 0.13 0.20 0.50 0.14 0.15 0.16 0.17 0.18 0.19
  t=3   0.19 0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.21 0.22
  t=4   0.22 0.19 0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.20
  t=5   0.30 0.31 0.32 0.33 0.34 0.60 0.25 0.70 0.10 0.40 0.20 0.15

solve(..., "discontinuous", ...)
  window_sizes   = [3, 3, 4, 4, 5, 5]   # dw = 3 + floor(t / 2), cds adds 2 per token
  cache_lens     = [7, 8, 8, 9, 10, 9]
  evict_steps    = [2, 5]               # t = 0, 1 only feed the jump gate
  kept_positions = [[0, 1, 2, 3, 7, 5, 9, 10, 11]]
  # t=5 keep=2 over [5, 6, 7, 8]: 0.70 then 0.60 -> positions 7, 5 in score order

solve(..., "slide", ...)
  window_sizes = [4, 4, 4, 4, 4, 4]  cache_lens = [7, 8, 8, 8, 8, 8]
  evict_steps  = [1, 2, 3, 4, 5]     kept_positions = [[0, 1, 2, 3, 7, 9, 10, 11]]

solve(..., "adaptive", ...)
  window_sizes = [3, 3, 4, 4, 5, 5]  cache_lens = [7, 7, 8, 8, 9, 9]
  evict_steps  = [0, 1, 2, 3, 4, 5]  kept_positions = [[0, 1, 2, 3, 7, 6, 9, 10, 11]]

solve(..., "None", ...)
  window_sizes = [0, 0, 0, 0, 0, 0]  cache_lens = [7, 8, 9, 10, 11, 12]  evict_steps = []
  kept_positions = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
```

```
tie break: P = 4, T = 3, H = 1, num_hidden_layers = 1, delta = 1
max_capacity_prompt = 2, decoding_recent_size = 1, metric = "adaptive"
attn rows, columns = positions 0..6
  t=0   0.40 0.10 0.30 0.20 0.15 0.25 0.35
  t=1   0.25 0.35 0.30 0.30 0.15 0.20 0.40
  t=2   0.05 0.45 0.20 0.30 0.40 0.40 0.10

window_sizes = [1, 2, 3]  cache_lens = [3, 4, 5]  evict_steps = [0, 1, 2]
kept_positions = [[0, 1, 4, 5, 6]]   # t=2 keep=2 over [4, 5], both score 0.40
```

## Constraints

- **The counters advance per layer call, not per token**: `cds` grows by exactly `num_hidden_layers` each token, so `dw = rec + floor(t / delta)`; incrementing it once per token makes the budget grow `num_hidden_layers` times too fast.
- **The burst gate's `jump` follows the same rule**: with `delta = 2` and 2 layers the pattern is "2 idle tokens, then 1 evicting token", so `evict_steps` lands on `t = 2, 5, 8, ...`.
- **The budget gate and the burst gate are in series**: `jump` only advances in calls where `L >= mcp + dw` — before that gate passes nothing accumulates and nothing evicts. That is exactly what separates `discontinuous` from "evict unconditionally every `delta + 1` steps".
- **`keep = dw - rec` may be 0**: `adaptive` still triggers eviction once `dw == rec`, but it empties the whole candidate region and keeps only the frozen prefix plus the trailing `rec` slots, dropping `cache_lens` to `mcp + rec` (that is `t = 0` in the second example).
- **`init_positions` keeps its slot order**: what is frozen is "the first `mcp` slots", not "positions below `mcp`" — a compressed cache is not position-ordered to begin with.
- Grading compares lists element-wise, not as sets, so `kept_positions` may read `..., 7, 5, 9, ...` in score order. `T` up to 400, `H` up to 8, `P` up to 512.
