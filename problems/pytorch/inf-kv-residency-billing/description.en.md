# KV Residency Integral and Token Billing Ledger

## Problem

An inference service bills along two unrelated axes:

- **Per token**: `output tokens` × unit price — the customer-facing number.
- **Per residency**: every KV block, from allocation to release, occupies so many "block-steps". This is the internal axis for cost, capacity planning, and batch-size limits, and the only way to tell where a long-tail request actually spends its money.

The hard part is not the pricing formula but the fact that **residency changes while the request generates**: it starts at `ceil(prompt_len / block_size)` blocks and gains one each time generation crosses a block boundary. So "block-steps" is a sum over time steps, not "blocks × steps".

## Task

Implement `solve(prompt_len, start, out_tokens, block_size, price_out, price_block_step)`:

- Inputs: `prompt_len` `(N,)`, `start` `(N,)` (the step it begins holding blocks), `out_tokens` `(N,)`, `block_size`, and the two prices (both `>= 0`).
- Request r holds `blocks(r, t) = ceil((prompt_len_r + (t - start_r)) / block_size)` blocks for global steps `t ∈ [start_r, start_r + out_tokens_r)`, and releases them immediately on exit.
- Outputs:
  - `residency` `(N,)` int64: the block-step integral per request;
  - `charge` `(N,)` float32: `price_out * out_tokens + price_block_step * residency`;
  - `peak_blocks` int64 scalar: the maximum simultaneous occupancy over the whole timeline;
  - `total_charge` float32 scalar.

## Example

```
prompt_len = [4, 10], start = [0, 1], out_tokens = [3, 2]
block_size = 4, price_out = 0.5, price_block_step = 0.1

r0: t=0 -> ceil(4/4)=1; t=1 -> ceil(5/4)=2; t=2 -> ceil(6/4)=2   integral = 5
r1: t=1 -> ceil(10/4)=3; t=2 -> ceil(11/4)=3                     integral = 6

timeline occupancy: t=0 -> 1, t=1 -> 2+3=5, t=2 -> 2+3=5, t=3 -> 0   peak = 5

charge(r0) = 0.5*3 + 0.1*5 = 2.0
charge(r1) = 0.5*2 + 0.1*6 = 1.6
total_charge = 3.6
```

## Constraints

- **Residency grows with time, so the final value cannot stand in for it**: `out_tokens × ceil((prompt_len + out_tokens) / block_size)` clearly overestimates, because early steps hold fewer blocks.
- **The interval is half-open**: `[start, start + out_tokens)`; nothing is held after the last token, and `out_tokens = 0` gives an integral of 0.
- **When `prompt_len` is an exact multiple of `block_size`**, the very first generated token needs one more block — the classic off-by-one.
- **The peak is simultaneous occupancy**, not a single request's maximum; overlapping requests add up.
- **`block_size` and the two prices must be positive/non-negative**; `out_tokens` may be 0.
