# Solution Notes · Automatic Prefix Caching

## Why the key must encode the whole prefix

A KV value depends not only on the tokens of its own block but on all tokens before it (attention is a function of position and history). Hence two blocks with identical content but different prefixes have completely different KV values and cannot be shared. Real vLLM uses `block_hash = hash(parent_block_hash, tuple(token_ids))` to chain the parent block's hash in, which is equivalent to this problem's "cumulative prefix tuple as key" approach — the key inherently carries the identity of the whole prefix.

## Hits must be contiguous

Because reuse means "skipping the prefill of an entire prefix span", as soon as block $j$'s prefix mismatches, block $j+1$ has a different prefix even if its content coincidentally matches → it cannot be reused. Therefore compare in order from block 0 and `break` at the first miss. With a trie representation, the hit count equals the maximum depth reachable along the path.

## Partial blocks are not counted

The trailing portion that does not fill `block_size` is not made into its own block and is not cached: KV block writes land on whole blocks, a half block has no stable hash identity, and other requests cannot align to it for exact reuse. Hence $n_\text{full}=\lfloor \text{len}/bs\rfloor$.

## What the hit rate means

For highly repetitive prefixes such as multi-turn conversations, shared system prompts, and few-shot templates, a hit directly skips the prefill of the corresponding tokens, significantly reducing TTFT and compute usage. The hit rate rises with the depth of the shared prefix; the `[0,2,...]` here is exactly the second request reusing the 2 full blocks already cached by the earlier requests.
