# Solution Notes · Suffix/REST Longest-Suffix Retrieval Draft

## Retrieval-based drafting = find the longest "tail we have seen before" for the context

REST/Suffix Decoding train no draft model; instead they maintain a datastore (historical prompt→continuation pairs, or a global stream of generated tokens), and at every decoding step take the current context and look inside for **the longest suffix that matches a contiguous substring of some record**, then copy the segment immediately following it in that record as the draft. Intuition: if the current tail has appeared in the corpus, it most likely continues in the same way again.

## Matching and tie-break

Let $n$ decrease from $\text{len(context)}$ down to $\text{min\_match}$; take the length-$n$ suffix $s$ of context and find the first occurrence of $s$ in each entry. The three tie-break priorities — **longer suffix > earlier entry > earlier position** — make the result unique:
- "longer first" treats the continuation of the longest tail as the most trustworthy draft; once some entry already yields a match no shorter than the current $n$, there is no need to look at smaller $n$ (you can break early);
- at equal length, scan entries in order and the first hit wins;
- within the same entry at the same length, scan $i$ in ascending order and the earliest occurrence wins.

The draft length is truncated to $\gamma$ and never runs past the end of the entry (if the match position is at the end of the entry, the draft is naturally empty).

## Division of labour with Lookahead(179)

| | Match unit | Output | Fallback |
|---|---|---|---|
| Lookahead | Suffixes of a fixed-length window $[\text{min\_n},\text{max\_n}]$, step by step | 1 token per step | Deterministic fallback exists |
| **Suffix/REST (this problem)** | **Variable-length longest suffix, in one shot** | **$\le\gamma$ draft tokens** | None (empty if nothing is found) |

## Edge-case self-checks

- `check_example`: the length-3 suffix `[1,2,3]` hits entry0, draft `[4,5]`; the shorter `[2,3]` is ignored.
- `check_no_match`: only a length-1 tail matches, but that is below `min_match` → empty draft.
- `check_tie_break_entry`: two entries hit at the same length → take the successor of the earlier entry.
- `check_earliest_position`: several hits inside the same entry → take the earliest position.
- `check_gamma_truncate`: the match position is near the end of the entry → the draft is truncated to the remaining length.
- `check_matches_oracle`: random datastore/context/gamma/min_match, cross-checked against an independent implementation that enumerates all (n,entry,pos) candidates and takes the maximum tuple.
