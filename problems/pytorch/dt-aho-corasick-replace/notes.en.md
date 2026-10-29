# Solution Notes: Aho–Corasick Multi-pattern Replacement

## Source and scope

Detection uses the classic Aho–Corasick algorithm introduced by Aho and Corasick in 1975: a trie, failure links, and output sets find multiple keywords together. **Leftmost priority, longest-at-that-start priority, smallest-index tie-breaking, and one-pass replacement** are explicit extensions defined by this problem. They are not presented as replacement semantics of the original paper or default behavior of any particular library.

## 1. Build the trie and retain duplicate indices

Each edge represents one Python character. Each node stores its children, failure link, and a list of pattern indices. Append the index at a pattern's terminal node; a Boolean terminal flag or deduplicated pattern string is insufficient. Duplicate patterns share a path but retain separate indices.

Validate empty patterns before any shortcut for empty text, raising `ValueError` when required.

## 2. Build failure links and inherited outputs with BFS

The root's direct children fail to the root. For each edge processed in BFS order, start from the parent's failure state and search for a transition with the same character. Follow failure links until a transition exists or the root is reached.

After assigning the child's failure link, append the failure target's outputs to the child's outputs. BFS ensures the shallower target already has its complete inherited output list, so every matching suffix is reported. Reporting only the current trie's terminal pattern would miss suffix matches.

In example 1, reaching the end of `she` reports its index 1 and both indices 0 and 3 of `he`.

## 3. Scan the original text once, then order the matches

Maintain an automaton state while reading characters. If a transition is missing, follow failure links. After consuming a character, emit every pattern in the current state's output list.

If $e$ code points have been consumed and a pattern has length $\ell$, its original span is $[e-\ell,e)$. Never insert replacement text during this scan. Sort the emitted tuples by `(start, end, index)` to obtain `all_matches`.

## 4. Select spans and construct the replacement separately

For selection, use a separate ordering by `(start, -end, index)`: at a fixed start, a larger end means a longer match. Keep the first record with `start >= consumed`, then set `consumed = end`. Do not reorder the required `all_matches` result in place.

This ordering implements the three contractual priorities exactly. After a selection, matches starting inside that span are ineligible, but adjacent matches remain eligible. This is neither globally longest-first selection nor an optimization of total covered length.

Finally, concatenate uncovered original slices and the indexed replacements from left to right, then append the original suffix. The actual output string for `ushers` is `uSrs`, with only `(1, 4, 1)` selected. The Unicode example produces `CE`, selecting `(0, 2, 1)`, `(2, 3, 0)`, and `(3, 5, 2)`. New adjacency caused by deletion must not trigger another replacement.

## Complexity

Let $P$ be total pattern length, $S$ the number of trie states ($S\le P+1$), $L$ the maximum pattern length (0 with no patterns), $H$ the total number of stored output entries after inheritance, $N$ the text length, $Z$ the number of matches, and $R$ the rewritten length.

Under amortized constant-time dictionary operations, this reference uses sparse transitions and explicit fallback while building BFS failure links. A conservative construction bound is $O(P+SL+H)$; copying inherited outputs must be included. Scanning takes $O(N+Z)$, the two sorts take $O(Z\log(Z+1))$, and string construction takes $O(N+R)$. Space including returned data is $O(S+H+Z+R)$. Empty inputs additionally incur constant overhead.

## Independent verification

The 15 `check_*` functions cover inherited outputs, multi-level fallback, suffixes and overlaps, duplicate ties, leftmost priority, longest priority, Unicode/combining characters, empty inputs, empty replacements, non-cascading behavior, immutability, and invalid empty patterns. Boundary cases use 500 characters and 30 patterns; 180 randomized cases use seed 8201.

The test oracle has no automaton: it enumerates occurrences using `startswith` at each original position. A separate position-advancing loop independently chooses the longest pattern at each current position and constructs the replacement, rather than reusing the reference solution's match-sorting and filtering structure. Tests also check exact Python container, tuple, and integer types.
