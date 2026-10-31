# Reference Approach: Position-Anchored Greedy Longest Match, Settled Per Word

## Implementation

Three nested loops, and the order cannot be rearranged:

1. **Word splitting**: maximal runs of `str.isspace()` characters separate words (equivalent to `text.split()`; the reference scans by hand while the oracle calls `split`, so the two cross-check each other). Word numbers increase from 0 in split order — an UNK neither consumes nor skips a number.
2. **Per-word greedy**: start at `start=0` and, at each `start`, shrink `end` from the right; the first hit is the longest hit. **The candidate string is the crux**: at `start == 0` it is the bare `word[start:end]`, at `start > 0` it must be `prefix + word[start:end]`. Using the bare form everywhere, or the prefixed form everywhere, are both wrong — and both still produce plausible-looking output on ordinary text.
3. **Settling per word**: the first `start` that comes back empty throws away every piece already produced for that word and leaves one `unk_token` spanning the whole word. So in `offsets` a `[UNK]` is always exactly a full word span, and `tokens` never shows "pieces followed by a mid-word UNK".

`max_span` (the longest vocabulary entry at word start, minus the prefix length for continuations) only shrinks the candidate endpoints; a longer candidate cannot be in the vocabulary, so results do not change. The mutant that drops this bound is equivalent — the judge cannot catch it and should not. The opposite direction (one too few) misses real hits and is caught.

Upstream interface, verbatim from the offline sources:

```text
# tokenizers 0.23.3, tokenizers/models/__init__.pyi:350-361
class WordPiece(Model):
    """
    An implementation of the WordPiece algorithm

    Args:
        vocab (:obj:`Dict[str, int]`, `optional`):
            A dictionary of string keys and their ids :obj:`{"am": 0,...}`

        unk_token (:obj:`str`, `optional`):
            The unknown token to be used by the model.

        max_input_chars_per_word (:obj:`int`, `optional`):
            The maximum number of characters to authorize in a single word.
    """

# tokenizers 0.23.3, tokenizers/implementations/bert_wordpiece.py
        wordpieces_prefix: str = "##",
    ):
        if vocab is not None:
            tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(unk_token)))
        ...
        tokenizer.decoder = decoders.WordPiece(prefix=wordpieces_prefix)
```

Measured defaults: `WordPiece(...).max_input_chars_per_word == 100` and `.continuing_subword_prefix == '##'`.

## Correctness checks

Ten `check_*` functions, all compared against a structurally independent oracle (the oracle buckets vocabulary entries by first character, sorts each bucket by descending length and takes the first successful `word.startswith(rest, start)`, which is the longest hit; the reference cuts endpoints and does dictionary lookups — no intermediate result is shared). On top of the comparison:

- `check_example`: hand-computed 4-tuples for the statement's example, including the second branch with the limit set to 3.
- `check_matches_oracle`: 24 fixtures covering multi-character prefixes, an empty prefix, `\t\n\u00a0` mixtures, CJK, combining sequences (`e` + U+0301), astral-plane emoji, a 40-code-point word, and an `unk_token` that is not at index 0 (this leaves nowhere for "hardcode the UNK id as 0" to hide).
- `check_word_level_unk_drops_pieces`: the statement's `hashcat` (hits `ha`, then `##sh`, then comes back empty) must keep only one `[UNK]`; additionally every `[UNK]` must be the sole piece of its word with a span exactly the word length — a per-character UNK answer explodes here.
- `check_greedy_is_not_minimum_pieces`: `ab/##c/##d` (three pieces) versus `a/##bcd` (two) — greedy takes three; then a reverse case where the long hit makes it one piece; finally `abcd` with `ab/##c/a/##bcd`, where greedy fails into UNK even though `a + ##bcd` would tile the word — this also kills "solve it with a DP over minimum pieces".
- `check_offsets_are_code_points`: astral characters count as 1 code point (2 UTF-16 units, 4 UTF-8 bytes) and CJK as 1 code point / 3 bytes; `_verify` additionally asserts `text[start:end]` equals the piece after stripping the prefix.
- `check_prefix_forms`: with an empty prefix bare entries are reusable mid-word; the multi-character `###` prefix; and a vocabulary entry that is exactly the prefix — unusable at `start > 0`, an ordinary entry at `start == 0`.
- `check_length_limit_boundary`: length equal to the limit passes, one code point over collapses the word, and the limit-of-1 case in both directions.
- `check_whitespace_and_numbering`: empty and whitespace-only text return four empty lists; word numbers stay contiguous.
- `check_invalid`: 22 rejected inputs plus 4 legal boundary calls.
- `check_maximum_shape`: 20000 vocabulary entries, text within 4000 code points, and a single 3999-code-point word.

`_verify` also enforces two general invariants — pieces of one word are contiguous and their lengths sum exactly to the word length, and `ids == [vocab.index(t) for t in tokens]` — plus that the input `vocab` was not mutated.

Mutation testing ran 17 mutants: per-character UNK, keep the partial pieces and append an UNK, bare lookup everywhere, look up with the prefix but emit it stripped, shortest match instead of longest, a minimum-piece DP, UTF-8 byte offsets, a UTF-16-unit length limit, one off the `max_span` bound, word-local offsets, `word_ids` filled with piece indices, the UNK id hardcoded to 0, three loosened validators (accept a bool limit, accept duplicate entries, accept a tuple vocabulary) and adding the prefix at word start — **16 caught**. The only survivor is dropping the `max_span` bound, which is bit-identical to the reference and merely slower.

## Scope and sources

- **The algorithm was compared word by word against the real implementation**: the reference solution was run against tokenizers 0.23.3's `WordPiece` in `.venv-eval` (with a `Whitespace` pre-tokenizer, no normalizer, `add_special_tokens=False`) — 16 vocabularies × 18 texts, 736 per-word comparisons, and **all 688 comparisons on words without `#` agree**. Every one of the 48 disagreements involves a word containing `#`; observed forms include `'a#' -> ['a', '[UNK]']` and `'##a' -> ['##', 'a']`, i.e. the Rust side has an extra branch around the prefix characters (and there it produces per-character UNK shapes). The Rust source is not in the offline environment, so **this problem does not reproduce that branch**: words containing `#` follow the stated rules, and that convention is defined by this problem.
- **Defined by this problem, no attribution claimed**: the code-point offsets and `word_ids` outputs (the upstream `Encoding` has its own offset semantics; here the rule is explicitly "indices into `text`'s code points"); `continuation_prefix` as an explicit argument (upstream it is the model's `continuing_subword_prefix` attribute); the vocabulary arriving as `list[str]` with the index as id (upstream it is `dict[str, int]`); raising `ValueError` for invalid input (upstream barely type-checks — `vocab` goes straight into Rust); and whitespace-only pre-tokenization — BertPreTokenizer's punctuation splitting and `BertNormalizer`'s lowercasing, accent stripping and CJK spacing are all out of scope.
- The default `max_input_chars_per_word` of 100 and the default `##` continuation prefix come from the offline sources and the measurements above; the `lowercase`, `clean_text` and `handle_chinese_chars` switches belong to `BertNormalizer` and are not involved here at all.
