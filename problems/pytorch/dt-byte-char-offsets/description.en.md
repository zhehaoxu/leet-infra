# UTF-8 Byte and Character Offset Alignment

## Task

Convert token intervals in the original text's UTF-8 byte stream into minimal character covers, then map between tokens, explicit pre-tokenization units and character queries. A “character” means a **Python Unicode code point**, not a UTF-16 unit or a grapheme cluster. Do not normalize text, insert prefixes, trim whitespace or mutate inputs.

## Inputs and validity

- `text`: a string containing only valid Unicode scalar values, with 0..100 code points.
- `word_spans`: a list of 0..30 `(start,end)` pairs. Each is a nonempty code-point interval within the text; spans are sorted by start and nonoverlapping. Whitespace gaps are allowed. These spans define the pre-tokenization units: do not split again on spaces. A unit may itself contain whitespace.
- `byte_offsets`: a list of 0..200 tuples `(a,b)`.
  - `(-1,-1)` exclusively denotes an inserted special token and may occur anywhere.
  - Every other interval must satisfy $0\le a<b\le |\operatorname{UTF8}(text)|$; otherwise raise `ValueError`. Empty, reversed or out-of-bounds intervals, and intervals with only one endpoint equal to -1, are invalid.
  - Ordinary endpoints may lie inside characters. Duplicate and overlapping intervals are allowed. Ignoring specials, starts are nondecreasing; ends need not be.
  - Every ordinary interval is guaranteed to lie entirely within one supplied word span's UTF-8 byte range.
- `char_queries`: a list of at most 100 Python integers, possibly negative or out of bounds.

Except for the ordinary-byte range requirement above, all inputs are guaranteed valid. You need not validate preconditions such as ordering or word containment. Every interval is half-open.

## Return value

Return a dictionary with exactly these four keys. Every field must be a **Python list**, and nested ranges must also be two-element lists, not tuples.

1. **`offsets`**: each ordinary token's minimal code-point cover. Round its start down to the owning code point and its end up to a code-point boundary. Endpoints already on boundaries stay there. Never decode partial UTF-8 bytes. A special token has `[0,0]`.
2. **`word_ids`**: the index in `word_spans` owning each ordinary token, even for partial-character bytes; `None` for specials.
3. **`char_to_token`**: for each query, the first token index covering that code point, in input token order. Return `None` if uncovered, including negative/out-of-bounds positions or gaps. Special tokens **never cover character 0**.
4. **`word_to_tokens`**: for each word, the smallest half-open token-index range containing all its tokens. For index set $I_w$, this is $[\min I_w,\max I_w+1)$; return `None` if the word has no tokens. The range may contain inserted specials.

## Executed examples

### 1. Multiple byte-level tokens can cover one character

```text
text = 'é😊X'
byte_offsets = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7)]
word_spans = [(0,1), (1,2), (2,3)]
char_queries = [0,1,2]

Output:
{
  'offsets': [[0,1], [0,1], [1,2], [1,2], [1,2], [1,2], [2,3]],
  'word_ids': [0,0,1,1,1,1,2],
  'char_to_token': [0,2,6],
  'word_to_tokens': [[0,2], [2,6], [6,7]]
}
```

`é` occupies 2 UTF-8 bytes, U+1F60A occupies 4, and `X` occupies 1. Repeated offsets are valid, and character queries return the first hit. This output was also verified against a complete local tokenizers 0.23.3 ByteLevel pipeline with a 256-symbol vocabulary and no merges.

### 2. Specials, a whitespace gap and word ranges

```text
text = 'é X'
byte_offsets = [(-1,-1), (0,1), (-1,-1), (1,2), (3,4), (-1,-1)]
word_spans = [(0,1), (2,3)]
char_queries = [0,1,2,-1,3]

Output:
{
  'offsets': [[0,0], [0,1], [0,0], [0,1], [2,3], [0,0]],
  'word_ids': [None,0,None,0,1,None],
  'char_to_token': [1,None,4,None,None],
  'word_to_tokens': [[1,4], [4,5]]
}
```

Word 0 owns token indices 1 and 3, so its range is `[1,4]`, including special token 2. Character 1 is a whitespace gap. Character 0 is first covered by token 1, never by special token 0.

## Scope

This is a restricted model of complete ByteLevel alignment with no normalization, prefix insertion or trimming, not an arbitrary-span interface to the standalone Rust `BytesToCharOffsetConverter`. Combining accents and ZWJ characters still count as separate code points. No UTF-16 or grapheme mapping, tokenizer implementation or BPE implementation is required.
