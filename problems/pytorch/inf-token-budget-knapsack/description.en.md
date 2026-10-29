# Per-Step Token Budget: 0/1 Knapsack for Prefill Admission and Chunked Leftover

## Problem

The number of tokens one forward pass can handle is the hardest budget in a serving engine: every decoding request needs at least one slot (otherwise its output stream breaks), and only what remains can go to prefilling new requests. Prompts differ in length and in priority, so deciding which requests to **fully** prefill in a single step is a weighted knapsack. Whatever the knapsack leaves over is usually handed to the request at the head of the queue as one more chunk (progressive admission in the spirit of chunked prefill).

This problem works out that single step's budget decision: given the waiting queue's prompt lengths and priority weights, the step budget and the number of decoding requests, output the admitted set, its total value, which request receives the leftover chunk, how many tokens it gets, and how much budget stays unused.

## Rules

Everything is integral. Within one step:

1. **Decode is the base load**: reserve `decode_count` slots first, leaving `remaining = budget - decode_count` slots for prefill (the inputs guarantee `budget >= decode_count`).
2. **0/1 knapsack**: choose an index set `S` with `Σ prompt_lens[i] <= remaining` for `i ∈ S`; every request in `S` completes its whole prefill this step. Compare candidates lexicographically by:
   - largest `Σ values[i]`;
   - ties: smallest `Σ prompt_lens[i]` (leaving budget for the next request);
   - further ties: the ascending index list of `S` is lexicographically smallest.
3. **Leftover chunk**: let `leftover = remaining - Σ_{i∈S} prompt_lens[i]`. If `leftover > 0` and some request is not in `S`, take the unadmitted request with the **smallest index** (the FCFS head) `j` and give it a chunk of `min(leftover, prompt_lens[j])` tokens — it is neither completed nor admitted this step. Whatever remains after that chunk is `unused`.

Return a five-tuple `(served, total_value, chunk_index, chunk_tokens, unused)`:

- `served`: ascending index list of the admitted set;
- `total_value`: sum of weights over the admitted set;
- `chunk_index`: index receiving the leftover chunk, `-1` if there is none;
- `chunk_tokens`: chunk size, `0` if there is none;
- `unused`: tokens still unspent after the chunk.

A zero-weight request is never admitted (it would only burn budget), but it can still receive the leftover chunk.

## Examples

**Example 1**: `prompt_lens = [5, 3, 8, 3]`, `values = [4, 2, 9, 1]`, `budget = 10`, `decode_count = 2`

```
remaining = 10 - 2 = 8
candidate sets (cost <= 8):
  {0, 1} cost 8  value 6
  {0, 3} cost 8  value 5
  {1, 3} cost 6  value 3
  {2}    cost 8  value 9    <- highest value
  ...
served = [2], exactly consumes the 8 slots, leftover = 0
```

```
served       = [2]
total_value  = 9
chunk_index  = -1
chunk_tokens = 0
unused       = 0
```

**Example 2** (zero weight, partially unused budget): `prompt_lens = [1, 2, 6]`, `values = [10, 0, 1]`, `budget = 10`, `decode_count = 0`

```
remaining = 10
{0, 2}     cost 7  value 11   <- optimal
{0, 1, 2}  cost 9  value 11   <- same value, larger cost, loses on key 2
leftover = 10 - 7 = 3
smallest unadmitted index is 1 (prompt = 2) -> chunk min(3, 2) = 2
unused = 3 - 2 = 1
```

```
served       = [0, 2]
total_value  = 11
chunk_index  = 1
chunk_tokens = 2
unused       = 1
```

**Example 3** (ties save budget): `prompt_lens = [2, 4]`, `values = [5, 5]`, `budget = 4`, `decode_count = 0`

```
remaining = 4
{0} cost 2 value 5;  {1} cost 4 value 5   -> tie on value, take the cheaper {0}
leftover = 4 - 2 = 2, chunk to unadmitted index 1: min(2, 4) = 2
```

```
served       = [0]
total_value  = 5
chunk_index  = 1
chunk_tokens = 2
unused       = 0
```

## Constraints

- At most 12 requests; `prompt_lens[i] >= 1`; `values[i] >= 0`; `budget >= 1`; `0 <= decode_count <= budget`; `budget <= 128`.
- At most one leftover chunk per step; it is never cascaded to further requests.
- `served` must be ascending and deduplicated; the other four items are Python `int`s.
- Only the return value is judged; any implementation is fine (capacity DP, memoised search, brute-force subset enumeration).
