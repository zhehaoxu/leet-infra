# Solution Notes · Suffix Automaton

## Why not a trie

Building one trie path per suffix yields \(1+2+\dots+N = O(N^2)\) nodes, and even storing only substrings stays \(O(N^2)\) in the worst case. SAM's key compression is **merging substrings with identical endpos**: identical endpos means they occur at exactly the same places in any context and therefore behave identically, so a whole family of consecutive-length substrings can live in one state.

A state therefore carries three things: `len` (the longest substring in the family), `link` (the next equivalence class up, representing the shorter suffixes) and `trans` (the transition table). The `link` chain is literally "keep shortening the current suffix", which is why the query side can walk back in amortized \(O(1)\).

## The three incremental moves

For each incoming character `c`:

1. Create `cur` with `len[cur] = len[last] + 1`;
2. Walk `link` upwards from `last`, pointing every state that lacks a `c` transition at `cur`;
3. Find the first ancestor `p` that already has a `c` transition to `q`:
   - if `len[p] + 1 == len[q]`, set `link[cur] = q`;
   - otherwise the new character **splits** `q`'s endpos set, so clone a state with `len = len[p]+1`, move part of `q`'s transitions and its link to the clone, then set both `link[q]` and `link[cur]` to the clone.

The split in step 3 is the one branch that is easy to miss. Omitting it inflates `distinct` (the same substrings counted twice) and shrinks the state count.

## Walking back on the query side

Keep `(state, best)`. On each character `c`:

- if `state` has a `c` transition: advance and `best += 1`;
- otherwise walk `link` upwards until a transition exists or the root is reached; on each step `best = len[state]` (the longest length of the new state is exactly the match length that survives);
- if even the root has no transition: `state = root`, `best = 0`.

`best` means "length of the longest suffix ending here that occurs in the corpus"—exactly the "how much suffix can I match" that suffix decoding wants. In a real system it sets the draft length, falling back to an n-gram pool or to no speculation when it is too small.

## Capacity and aging

SAM never evicts states (the state count is bounded by \(O(N)\), but \(N\) keeps growing on long contexts). In practice REST / Suffix Decoding rebuild periodically or shard by hash and keep capacity management outside the automaton, so this problem covers the automaton itself while capacity and aging are tested separately in `spec-ngram-pool-aging`.

## Related problems

- `spec-suffix-draft`: longest-suffix retrieval over a **static datastore**, without incremental construction.
- `spec-ngram-pool-aging`: a capacity-limited n-gram pool focusing on eviction and aging policies.
- `spec-lookahead-jacobi`: another model-free draft source, using n-gram matching with Jacobi iteration.
