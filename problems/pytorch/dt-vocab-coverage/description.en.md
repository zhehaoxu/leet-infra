# Vocabulary Coverage and OOV Statistics

## Task

Given tokenized `documents`, a vocabulary `vocab`, and `ignored_tokens`, compute occurrence counts, document frequencies, token coverage, and type coverage for the retained corpus.

First remove token occurrences **exactly equal** to an ignored token, then compute every statistic. Do not lowercase, normalize Unicode, or match substrings. Keep out-of-vocabulary (OOV) tokens distinct rather than collapsing them into one unknown ID.

## Inputs and constraints

- `documents` is a `list[list[str]]` containing 0..30 documents, each with 0..100 tokens.
- Every token is a nonempty Unicode string of length 1..30, measured in Python code points.
- `vocab` is a list of 0..100 unique tokens. **Raise `ValueError` for duplicates**, even if the duplicated token is ignored or the corpus is empty. All other inputs are guaranteed valid.
- `ignored_tokens` is a token list allowing duplicates; repeated ignored entries have no additional effect.
- Preserve document order. Do not discard empty documents or documents emptied by filtering.
- Do not mutate any input, including inner document lists. Only the Python standard library is needed; tensors are unnecessary.

## Return value

Return a Python `dict` with exactly these eight keys. Dictionary insertion order is not constrained.

| Key | Python type | Meaning |
| --- | --- | --- |
| `vocab_counts` | `list[int]` | Retained corpus occurrence counts, aligned with the original `vocab` order |
| `vocab_doc_freq` | `list[int]` | Number of retained documents containing each vocab token, counting each document at most once per token; aligned with `vocab` |
| `oov_tokens` | `list[str]` | Distinct retained OOV tokens in the order specified below |
| `oov_counts` | `list[int]` | Corpus occurrence counts aligned with `oov_tokens` |
| `oov_doc_freq` | `list[int]` | Document frequencies aligned with `oov_tokens`, counting each document at most once per token |
| `document_stats` | `list[tuple[int, int]]` | `(retained occurrence count, OOV occurrence count)` for each document in input order |
| `token_coverage` | `float` | Retained in-vocab occurrences / all retained token occurrences |
| `type_coverage` | `float` | Observed retained in-vocab types / all observed retained types |

Additional rules:

- Sort OOV tokens by `(-corpus_count, token)`: decreasing occurrence count, then increasing Python string lexicographic order, not locale-aware or numeric order.
- Ignored or unseen vocabulary entries remain in the aligned arrays with zero counts and document frequencies.
- `document_stats` counts **occurrences**, not unique types. An empty document contributes `(0, 0)`.
- Each coverage uses its own denominator. Return `0.0` if that denominator is zero. Unobserved vocabulary entries do not contribute to the type denominator. Do not additionally round coverage values.
- All counts must be Python `int` values. Use the Python built-in containers and numeric types shown in the table.

## Example 1: ignored tokens and empty documents

Input:

```python
documents = [['a', 'x', 'a'], [], ['x', 'b']]
vocab = ['b', 'a', 'z']
ignored_tokens = ['b']
```

Output:

```python
{
    'vocab_counts': [0, 2, 0],
    'vocab_doc_freq': [0, 1, 0],
    'oov_tokens': ['x'],
    'oov_counts': [2],
    'oov_doc_freq': [2],
    'document_stats': [(3, 1), (0, 0), (1, 1)],
    'token_coverage': 0.5,
    'type_coverage': 0.5
}
```

After ignoring `b`, four occurrences remain, two of which are the known token `a`. The two observed types are `a` and `x`, with one in the vocabulary. Although `a` occurs twice, it appears in only one document.

## Example 2: exact Unicode matching and different denominators

Escapes have their Python string-literal meanings: `\xe9` is `é`, `\xc9` is `É`, and `\u732b` is `猫`. The string `e\u0301` is not normalized to `é`.

Input:

```python
documents = [
    ['\xe9', 'e\u0301', '\xc9', '\u732b', '\U0001f642', '\U0001f642'],
    ['\xe9', '\xc9', '\U0001f642']
]
vocab = ['\xe9', 'unused']
ignored_tokens = ['\xc9']
```

Output:

```python
{
    'vocab_counts': [2, 0],
    'vocab_doc_freq': [2, 0],
    'oov_tokens': ['\U0001f642', 'e\u0301', '\u732b'],
    'oov_counts': [3, 1, 1],
    'oov_doc_freq': [2, 1, 1],
    'document_stats': [(5, 4), (2, 1)],
    'token_coverage': 0.2857142857142857,
    'type_coverage': 0.25
}
```

There are seven retained occurrences, two known, so token coverage is $2/7$. One of four observed types is known, giving type coverage $1/4$. The two OOV types with one occurrence each are ordered lexicographically by Python strings.
