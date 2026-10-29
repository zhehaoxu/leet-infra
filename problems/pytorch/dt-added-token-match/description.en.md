# Added-Token Isolation and Longest Matching

## Task

Implement `solve` to isolate registered added tokens in ASCII source text while retaining all ordinary text and offsets. This is a restricted, single-stage model with identity normalization and no whitespace stripping or BPE.

## Input

- `text`: an ASCII string of length 0 to 300.
- `added_tokens`: a registration list of length 0 to 30. Each dictionary has exactly:
  - `content`: nonempty, whitespace-free ASCII, at most 30 characters.
  - `single_word`: a boolean enabling neighbor checks after raw candidate selection.
  - `special`: a boolean marking a candidate that can be skipped.
- `split_special_tokens`: a boolean, default `False`.
- Empty or duplicate contents must raise `ValueError`, even for empty text or entries that special filtering would remove. All other inputs are valid.
- Do not mutate the registration list or its dictionaries.

## Exact matching rules

1. **Filter special patterns first**: with `split_special_tokens=True`, remove every `special=True` pattern from the candidate set before matching. Other patterns remain eligible and retain their original registration indices.
2. **Select nonoverlapping raw candidates**: choose the occurrence with the earliest start; among occurrences with that start, choose the longest content. Continue from the selected interval's end. This is not globally longest matching or registration-order selection of a shorter string.
3. **Check boundaries afterward**: when the selected candidate has `single_word=True`, its immediate neighbors, if present, must not be ASCII letters `A-Z/a-z`, digits `0-9`, or `_`. Only the neighbors are checked; content itself need not start or end with a word character.
4. A longest candidate that fails the boundary check becomes ordinary text but still consumes its whole raw interval. **Never fall back to a shorter pattern or rescan inside it**, including matches that start inside and extend beyond the rejected interval.
5. Matching is case-sensitive. Special tokens have no inherent priority. All patterns share one stage; there is no numeric priority, normalization, BPE, or `lstrip` / `rstrip` whitespace absorption.

The problem's “remove special patterns first” rule differs from native tokenizers 0.23.3, which skips special matches after candidate selection. Follow this problem's contract; see the notes for the verified distinction.

## Output

Return a list of `(piece, start, end, added_id)` tuples:

- `piece` is an original-text slice; offsets are half-open Python character indices.
- Emit every accepted added token separately, with its **original registration index**.
- Merge all contiguous unmatched source text, including rejected candidates, into ordinary pieces with ID `-1`.
- Adjacent accepted matches stay separate even when they have the same ID. Emit no empty pieces.
- Pieces partition the source text. Empty text returns `[]`, but registration validation must still run first.

## Example 1: leftmost, not globally longest

```python
text = 'abcde'
added_tokens = [
    {'content': 'ab', 'single_word': False, 'special': False},
    {'content': 'bcde', 'single_word': False, 'special': False},
]
split_special_tokens = False
output = [('ab', 0, 2, 0), ('cde', 2, 5, -1)]
```

`ab` starts earlier, blocking the overlapping `bcde` occurrence.

## Example 2: no fallback after boundary rejection

```python
text = 'zabc!'
added_tokens = [
    {'content': 'ab', 'single_word': False, 'special': False},
    {'content': 'abc', 'single_word': True, 'special': False},
    {'content': 'b', 'single_word': False, 'special': False},
]
split_special_tokens = False
output = [('zabc!', 0, 5, -1)]
```

Select `abc` first, then reject it because its left neighbor is `z`. Do not fall back to `ab` or rescan the internal `b`.

## Example 3: pre-removing special patterns changes selection

```python
text = 'abc!'
added_tokens = [
    {'content': 'ab', 'single_word': False, 'special': False},
    {'content': 'abc', 'single_word': True, 'special': True},
]
split_special_tokens = True
output = [('ab', 0, 2, 0), ('c!', 2, 4, -1)]
```

For the same input with the flag set to `False`, the result is:

```python
[('abc', 0, 3, 1), ('!', 3, 4, -1)]
```

## Implementation requirements

Use only the Python standard library. A trie can select the longest raw candidate at each scan position, followed by a separate boundary check. Full AddedVocabulary behavior is not required.
