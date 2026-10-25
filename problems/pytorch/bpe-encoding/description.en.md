# Tokenization: Encoding Text with BPE Rules

## Problem

Given already-learned BPE merge rules (the output of `bpe-training`), split a word into subwords. This is precisely the first step of turning raw text into a token sequence at inference time. The core rule: **each time apply the currently available merge pair with the highest priority (earliest in the learning order)** until no available pair remains.

## Algorithm (be sure to implement exactly this)

Given a word `word` (a lowercase string) and the ordered merge rules `merges` (`[(a, b), ...]`, where the index is the priority rank, smaller applied first):

1. Split into characters and append `</w>` at the end: `"newer"` → `["n","e","w","e","r","</w>"]`.
2. Record `rank[(a,b)] = the index of that pair in merges`.
3. Repeat: among all adjacent bigrams of the current symbol sequence, pick those **present in `merges`** and take the pair with the **smallest rank** (earliest); replace all **left-to-right, non-overlapping** occurrences of that pair in the word with the concatenated symbol `a+b`.
4. Stop when no adjacent bigram matches `merges`, and return the current symbol sequence (that is, the subword list).

> Note that this is "re-select the globally smallest-rank available pair every round", not "scan through merges once from the start" — because earlier merges create new mergeable pairs.

## Rules

Implement `solve(word, merges)`, returning the subword list `list[str]` (including the trailing `</w>`).

## Examples

With `merges = [('e','r'), ('er','</w>'), ('l','o'), ('lo','w'), ('e','w'), ('ew','er</w>'), ('n','ewer</w>'), ('low','</w>')]`:

```
"newer" → ["newer</w>"]
"lowest"→ ["low", "e", "s", "t", "</w>"]
"wider" → ["w", "i", "d", "er</w>"]
"low"   → ["low</w>"]
```

In `"lowest"`, `l o w` is merged into `low`, but the remaining `e s t` can no longer be merged (`('e','s')`, `('s','t')` are not in merges), so it stops at 5 segments.

## Constraints

- `word` contains only lowercase letters; `merges` may be an empty list (then return the characters plus `</w>` unchanged)
- Pure string/dict operations, no randomness; replacement must be **non-overlapping, left to right**
- Echoes `bpe-training`: training produces merges and encoding consumes the same merges
