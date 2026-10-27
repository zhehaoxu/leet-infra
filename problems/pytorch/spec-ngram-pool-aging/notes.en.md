# Solution Notes · n-gram Pool Aging

## Where each of the three parts goes wrong

- **Predict**: `max(table.values())` gives a count, not a token. Sort by "count descending, token ascending" together; otherwise ties depend on dict iteration order and the judged result wobbles with the implementation.
- **Learn**: the tempting shortcut is "add 1 to the token I predicted". That lets wrong guesses self-amplify into a loop (predict A → A's count grows → predict A more), producing a beautiful hit rate with collapsing draft quality. Always update with the true token.
- **Evict**: plain LRU is the usual simplification but conflicts with "frequent first": an n-gram seen 50 times can be pushed out by a brand-new key seen once just because it was used 3 steps ago. The triple `(total count, timestamp, key)` stacks both goals: count decides survival, the timestamp only breaks ties.

## Why the timestamp must be globally monotone

If timestamps were per-step ordinals or per-key counts, ages would not be comparable across keys. With one global monotone clock, `stamp[k]` orders exactly as "who has gone unused longer", which makes eviction and aging single-valued and keeps judging reproducible.

## Initial trimming

`prompt` may hold thousands of tokens, far exceeding `capacity` once sliced. If trimming only happened after queries, the first frame's `size` would report the raw entry count (fake inflation) and the first trim would dump hundreds of keys at once, distorting the whole `evicted` sequence. So trim immediately after building the initial pool.

## Boundaries

- `capacity = 1`: learning any new key evicts the incumbent, so the pool always holds one key. If that key is frequent it survives indefinitely—count-first behavior in its purest form.
- `n = 1` with a short `prompt`: only one or two keys, almost everything hits; larger `n` makes the pool sparser, lowering the hit rate but relying more on context.
- Empty `observed`: all four outputs are zero-length tensors while the pool is still built (just with no observation points for `size`/`evicted`).
- A miss means "the key is not in the pool", which is different from "the key is there but the successor is missing"—the latter cannot happen because every insertion carries a successor.

## Complexity

One hash lookup plus one \(O(N)\) trim per step (\(N\) = pool size); a min-heap with lazy deletion can bring trimming to \(O(\log N)\). Pool sizes in real systems are in the thousands, so a linear scan is perfectly acceptable.

## Related problems

- `spec-suffix-automaton`: does the same "history suffix matching" exactly with \(O(N)\) states, without capacity eviction.
- `spec-lookahead-jacobi`: the upstream use of such a pool—how matched n-grams become parallel drafts.
- `spec-suffix-draft`: longest-suffix retrieval over a static datastore.
