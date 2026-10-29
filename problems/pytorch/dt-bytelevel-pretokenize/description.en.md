# GPT-2 Byte-Level Pretokenization

## Task

Implement `solve`: split source text using the fixed GPT-2 rules, then encode each piece through a reversible UTF-8 byte-to-Unicode mapping. There is no normalization, added prefix space, or BPE merging.

## Input and output

- `text` contains 0 to 300 valid Unicode scalars defined in Python 3.12 / Unicode 15.0. Surrogates and unassigned characters are excluded; characters introduced by later Unicode versions are out of scope.
- Return a list of `(byte_encoded_piece, start, end)` tuples.
- `start` and `end` are half-open **original Python code-point offsets**, not UTF-8 byte offsets or positions in the encoded string.
- Nonempty pieces partition the original text in order. Empty text returns `[]`.

## Splitting rules

This exact, ordered, case-sensitive pattern defines the semantics. Implement it with standard-library `unicodedata.category` and a handwritten ordered scanner, not a third-party regex engine:

```text
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

1. Lowercase contractions take priority only at the current scan position. `'S` is not `'s`.
2. `L` means a Unicode category starting with `L`. `N` means a category starting with `N`, including `Nd`, `Nl`, and `No`, not just decimal digits.
3. Combining marks, emoji, punctuation, and all other non-whitespace, non-letter, non-number characters form the third class. Do not attach combining marks to letters automatically.
4. The optional leading space on these three run types is at most one ASCII `U+0020`. A punctuation run can consume a later apostrophe; do not restart contraction matching inside that run.
5. Whitespace is exactly:

```text
U+0009–U+000D, U+0020, U+0085, U+00A0, U+1680,
U+2000–U+200A, U+2028–U+2029, U+202F, U+205F, U+3000
```

`U+001C–U+001F` are not members, so `str.isspace()` is not a substitute. At a whitespace branch, consume an entire trailing run. For an internal run longer than one character, consume all but its last character and handle that last character on the next iteration. A single internal whitespace uses the final branch.

## Byte encoding

UTF-8 encode each source piece, then map every byte:

- Bytes `33–126`, `161–172`, and `174–255` map to Unicode characters with the same numeric value.
- All remaining bytes, in ascending byte order, map to consecutive characters beginning at `U+0100`.

For example, ASCII space becomes `Ġ` and newline becomes `Ċ`. The encoded string is a reversible byte representation, not a translation.

## Example 1: original offsets

```python
text = '你好  café\n'
output = [('ä½łå¥½', 0, 2), ('Ġ', 2, 3), ('ĠcafÃ©', 3, 8), ('Ċ', 8, 9)]
```

`你好` occupies 2 source code points even though its encoded piece has 6 characters. Of the two middle spaces, one forms a separate whitespace piece and the other prefixes the letter run.

## Example 2: case-sensitive contractions

```python
text = "We're I'M can't"
output = [('We', 0, 2), ("'re", 2, 5), ('ĠI', 5, 7), ("'", 7, 8), ('M', 8, 9), ('Ġcan', 9, 13), ("'t", 13, 15)]
```

## Example 3: greedy punctuation

```python
text = "!!'s 's"
output = [("!!'", 0, 3), ('s', 3, 4), ("Ġ'", 4, 6), ('s', 6, 7)]
```

The first punctuation run consumes the apostrophe. The later match starts at a space and takes “space plus punctuation.” Neither position produces a separate `'s` contraction piece.

## Implementation requirements

Use only the Python standard library. Do not import `tokenizers` or third-party `regex`, and do not convert offsets to byte offsets. A linear ordered scanner is sufficient; a full tokenizer is not required.
