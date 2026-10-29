# Aho–Corasick Multi-pattern Replacement

## Task

Given an original string `text`, a list `patterns`, and a corresponding list `replacements`, find every pattern occurrence in the original text. Then select non-overlapping spans by a leftmost-longest rule and replace them exactly once.

Implement multi-pattern detection with an **Aho–Corasick trie, BFS-built failure links, and inherited outputs**, rather than trying every pattern at every text position.

## Inputs and constraints

- `text`: a Python `str` containing 0..500 Unicode scalar characters.
- `patterns`: a list of 0..30 strings, each of length 1..30. Duplicate patterns are allowed.
- `replacements`: an equally long list of strings; individual replacements may be empty.
- **Raise `ValueError` if any pattern is empty**, even when the text is empty. All other inputs are guaranteed valid.
- Lengths and offsets count Python Unicode code points, not UTF-8 bytes or grapheme clusters. Do not change case or normalize Unicode.
- Do not mutate any input. Only the Python standard library is needed; tensors are unnecessary.

## Output and rules

Return the Python tuple `(rewritten, all_matches, selected)`:

1. `rewritten` is the resulting Python string.
2. `all_matches` is a list of Python tuples `(start, end, pattern_index)`. All indices are zero-based, and `[start, end)` is a half-open span in the **original** `text`. Include all occurrences, including overlaps, suffix matches, and separate entries for each index of a duplicate pattern. Sort by `(start, end, pattern_index)` ascending.
3. `selected` is a list of the same tuple type, in greedy selection order:
   - Start with consumed position 0; only matches starting at or after that position are eligible.
   - Choose the smallest `start`, then the largest `end - start`, then the smallest `pattern_index`.
   - Advance the consumed position to the chosen match's `end`. Skip every match with `start < end`; a match starting exactly at `end` remains eligible.
   - Repeat until no eligible match remains. A longer match starting later cannot take precedence over an earlier match.

Replace each selected span with `replacements[pattern_index]`, preserving uncovered original text exactly. **Never rescan** replacement strings or new matches created by joining pieces. All spans continue to refer to the original text. Every tuple index must be a Python `int`; do not substitute tensors or other container types.

With no patterns or no matches, return the unchanged text and two empty lists.

## Example 1: failure outputs, duplicates, and leftmost priority

Input:

```python
text = 'ushers'
patterns = ['he', 'she', 'hers', 'he']
replacements = ['H', 'S', 'R', 'D']
```

Output:

```python
('uSrs', [(1, 4, 1), (2, 4, 0), (2, 4, 3), (2, 6, 2)], [(1, 4, 1)])
```

At the end of `she`, both indexed copies of its suffix `he` also match. Although `hers` is longer, it starts later, so `she` at `[1, 4)` is selected. The original prefix `u` and suffix `rs` remain.

## Example 2: combining characters, Unicode offsets, and deletion

Escapes below have their Python string-literal meanings: `\xe9` is `é`, `\u0301` is a combining accent, and `\U0001f642` is one supplementary-plane character.

Input:

```python
text = 'e\u0301\xe9\U0001f642\xe9'
patterns = ['\xe9', 'e\u0301', '\U0001f642\xe9']
replacements = ['E', 'C', '']
```

Output:

```python
('CE', [(0, 2, 1), (2, 3, 0), (3, 5, 2), (4, 5, 0)], [(0, 2, 1), (2, 3, 0), (3, 5, 2)])
```

The `e` plus combining accent occupies two code points, while precomposed `é` occupies one. They are not normalized into the same pattern. The last selected span `[3, 5)` is deleted by its empty replacement.
