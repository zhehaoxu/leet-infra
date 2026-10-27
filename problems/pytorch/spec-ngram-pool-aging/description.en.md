# Speculative Decoding: Capacity Eviction and Aging in an n-gram Pool

## Problem

Model-free draft sources—lookahead decoding, prompt lookup decoding, n-gram speculation—all rely on a table harvested from the corpus: use the last \(n\) tokens as a key and look up the most common (or most recent) successor as the draft. As generation grows longer the table grows without bound, so in practice it needs a **capacity limit** and an **eviction rule**. Getting that rule wrong has a direct cost: evicting frequent, stable n-grams makes the hit rate collapse, while evicting purely by recency means sparse but correct new n-grams never get in.

This problem splits the decision into three parts, all of them checkable integer/boolean behavior:

1. **Predict**: on a hit, take the most frequent successor under that key; break ties with the smaller token id (for determinism).
2. **Learn**: on a miss, insert that n-gram using the **true token** observed at this step—reinforce the real token, never your own guess, otherwise mistakes get frozen into high counts.
3. **Evict**: whenever the key count exceeds `capacity`, remove the entry minimizing `(total count, last-used timestamp, key lexicographic order)`. **Total count first** keeps frequent n-grams from being pushed out by one-off new keys; the timestamp ages out stale entries with equal counts.

Timestamps come from a single global monotone clock: the initial corpus is inserted in position order, and every later insert/reinforce increments it. That makes "older" always unambiguous.

## Task

Implement `solve(prompt, observed, n, capacity)`:

- Input: `prompt` (`(L,)` int64, the initial corpus, \(L\ge n+1\)); `observed` (`(S,)` int64, the true tokens revealed step by step, may be empty); `n` (key length, \(\ge 1\)); `capacity` (key limit, \(\ge 1\)).
- Flow: slice all n-grams out of `prompt` and trim to capacity, then for each entry of `observed` do "lookup-predict → reinforce or insert with the true token → trim".
- Outputs:
  - `predicted` (`(S,)` int64): the predicted token on a hit, `-1` on a miss;
  - `hits` (`(S,)` bool): whether the step hit;
  - `size` (`(S,)` int64): number of keys after trimming that step;
  - `evicted` (`(S,)` int64): how many keys were evicted that step.
- Raise `ValueError` when `n < 1`, `capacity < 1`, `prompt` is shorter than \(n+1\), or any input is not 1-D.

## Example

```
prompt = [1,2,3,1,2,3], n = 2, capacity = 4
initial pool: (1,2)->{3:2}   (2,3)->{1:1}   (3,1)->{2:1}       size = 3

observed = [3, 1, 9, 3, 4]

step 0  key=(2,3) hit  -> predict 1; reinforce (2,3)[3], size 3
step 1  key=(3,3) miss -> -1; insert (3,3)->1, size 4
step 2  key=(3,1) hit  -> predict 2; reinforce (3,1)[9], size 4
step 3  key=(1,9) miss -> -1; after inserting (1,9)->3 size 5 > 4
        -> evict (3,3) (total count 1, oldest timestamp)  size 4, evicted 1
step 4  key=(9,3) miss -> -1; after insertion size 5 > 4
        -> evict (1,9) (total count 1, next oldest)       size 4, evicted 1

predicted = [1, -1, 2, -1, -1]
hits      = [T,  F, T,  F,  F]
size      = [3, 4, 4, 4, 4]
evicted   = [0, 0, 0, 1, 1]
```

## Constraints

- **Reinforce with the true token**: on a hit, count the `observed` token under that key. Reinforcing your own prediction lets wrong guesses self-amplify—the hit rate climbs while quality falls.
- **Eviction is a triple comparison**: smallest total count first, then the older timestamp, then the lexicographically smaller key. Comparing timestamps alone degenerates to pure LRU, where one cold step evicts a frequent n-gram.
- **Trim once after initialization too**: the initial corpus may far exceed capacity, so the first frame's `size` cannot be the raw entry count.
- **Missed keys are inserted as well**: this problem treats a miss as a learning opportunity, equivalent to updating the n-gram table online; otherwise new contexts can never enter the pool.
- **`predicted` is `-1` on a miss**, distinct from "predicted token id -1"; token ids here are non-negative.
