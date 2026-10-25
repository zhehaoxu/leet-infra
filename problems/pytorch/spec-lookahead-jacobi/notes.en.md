# Solution Notes · Lookahead's Jacobi n-gram Pool

## Replacing the draft with "looking up historical n-grams"

Lookahead has no draft model; it views language modeling as a set of equations that can be iterated in parallel à la Jacobi: guess several consecutive positions and fill those guesses with **n-gram continuations that have been seen before**. Reduced to a single-stream implementation, the core is a table `pool: n-gram -> next token`:
- In the seeding phase, every substring of length $\in[\text{min\_n},\text{max\_n}]$ of the `pool_seed` sequences is mapped to its next token;
- At generation time the **longest suffix** of the current $seq$ is used to query the table, and a hit yields the speculative `drafted`;
- If nothing is found, fall back to the deterministic token $(seq_{-1}+1)\bmod V$ (in a real implementation this step is one forward pass of the target model; here a deterministic function replaces it so that it can be graded).

## Learning while generating

Every time a token is appended, the "n-gram in front of it → it" entry is written back into the pool, so the pool keeps growing and reinforcing itself during generation. Two ordering details must be fixed:
1. **Suffix lookup** goes from `max_n` down to `min_n`—prefer longer context and prevent short n-grams from stealing the hit;
2. **Pool update** goes from `min_n` up to `max_n`—a single update writes keys of several window lengths, so the order determines which overwrite wins and must match the oracle exactly.

Also note **lookup before write**: the draft of the current step depends only on the pool as it was before the write, which guarantees that a token never serves both as query context and as its own draft.

## Differences from other speculative problems

Lookahead / REST / n-gram are all "retrieval-based drafts" (no neural draft model). Lookahead uses **fixed-length window suffixes + a self-learning pool + deterministic fallback**; REST (problem 180) instead uses **variable-length longest-suffix matching** and performs no per-token fallback. Both emphasize lightweight speculation without a draft model.

## Edge-case self-checks

- `check_example`: a fixed-length 2-gram pool, two hits followed by two fallbacks (chain `[0,1,2,3]` → `[2,3,4,5]`).
- `check_empty_pool_fallback`: empty pool → everything goes through the $(last+1)\bmod V$ fallback.
- `check_seed_override`: an n-gram with the same key is overwritten by a later seed, verifying "later writes win".
- `check_matches_oracle`: random prompt / window widths / seeds / vocabulary, matched step by step against an independent pure-Python implementation.
