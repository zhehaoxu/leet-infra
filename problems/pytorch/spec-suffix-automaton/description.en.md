# Speculative Decoding: Online Suffix Automaton and Longest Suffix Match

## Problem

Suffix Decoding / REST style methods treat "suffixes seen in history" as a free source of drafts: given the current context, find the **longest matching suffix** in history and use the tokens that followed it as the draft. Doing this online (pushing new tokens into the index while generating, and querying at any time) needs an incremental structure, and the suffix automaton (SAM) is the standard answer: \(O(N)\) states, \(O(N)\) construction, \(O(P)\) queries.

What SAM really does is not "store the substrings" but **compress substrings by endpos equivalence classes**: two substrings with the same endpos (the set of end positions of all their occurrences) always appear together in every context, so they can be merged into one state. Consequently:

- A state \(s\) represents a family of substrings with consecutive lengths in \((\text{len}[\text{link}[s]],\ \text{len}[s]]\);
- the number of distinct substrings is \(\sum_{s\ne \text{root}} \big(\text{len}[s]-\text{len}[\text{link}[s]]\big)\);
- the state count never exceeds \(2N-1\) (the bound for the minimal DFA).

Construction is incremental: for each new character, create a state and add transitions along suffix links; when a transition points at a state whose length does not fit, **clone** it—the only genuinely tricky part, and the reason this problem exercises repetitive strings.

The query side is incremental too: keep "current state + current match length"; on each character either follow a transition, or walk suffix links back to an ancestor that has one (shrinking the match length to `len[link]`), and reset to zero when even the root cannot move. Each step's answer is therefore "the longest suffix ending here that is a substring of the text".

## Task

Implement `solve(text, pattern)`:

- Input: `text` (`(N,)` int64, the corpus, \(N\ge 1\)); `pattern` (`(P,)` int64, the query, may be empty).
- Outputs:
  - `n_states` (int64 scalar): number of automaton states (including the initial state);
  - `distinct` (int64 scalar): number of distinct substrings of `text`;
  - `match_len` (`(P,)` int64): `match_len[i]` is the length of the longest suffix of `pattern[:i+1]` that is still a substring of `text`.
- Raise `ValueError` when `text` is empty or either input is not 1-D.
- Complexity: \(O(N)\) construction and \(O(P)\) queries; brute-force enumeration of suffixes per position is not acceptable.

## Example

```
text    = [1, 2]
pattern = [1, 2, 1, 3]

States after construction (root is S0):
  S0: len 0
  S1: len 1, link S0   ("1")
  S2: len 2, link S0   ("2" / "12" merged into one family: length range (0, 2])

n_states = 3
distinct = (1-0) + (2-0) = 3        # "1", "2", "12"

Query:
  "1"    -> transition exists,      match_len = 1
  "12"   -> keep advancing,         match_len = 2
  "121"  -> walk suffix links back, match_len = 1   ("1" is a substring, "21"/"121" are not)
  "1213" -> back at root, no transition, match_len = 0

match_len = [1, 2, 1, 0]
```

## Constraints

- **State count is not substring count**: S2 represents both "2" and "12" because both have endpos `{2}`; a trie with one state per substring degenerates to \(O(N^2)\) nodes.
- **Cloning must happen**: when a new character splits a state's endpos set, you must clone a shorter state and redirect part of the transitions, otherwise the state count inflates and `distinct` is wrong. This problem uses `[1,2,2]` to force a split (5 states > n+1).
- **`match_len` shrinks with the link walk**: after jumping to state \(t\), the match length becomes exactly `len[t]`, not some value reduced by a constant.
- **The query may be empty**: return an empty tensor for an empty `pattern`, but still report `n_states`/`distinct`.
- The vocabulary is unbounded (token ids can be arbitrarily large), so use dictionary transitions rather than an alphabet-sized array.
