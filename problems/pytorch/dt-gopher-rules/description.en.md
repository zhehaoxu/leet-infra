# Gopher Rule-Based Quality Filter: Ordered Ledger of Nine Heuristics

## Problem

Before training its 280B model, Gopher (DeepMind, arXiv:2112.11446) cleaned its corpus with a set
of **rule-based heuristics**: no model, no learned parameters — just nine interpretable threshold
rules that drop low-quality documents one by one. It remains the standard baseline of data
pipelines, and HuggingFace's datatrove ships the authoritative `GopherQualityFilter`. The rules are
simple; the **ledger is hard to align**: which denominator each rule uses (all words or
non-punctuation words?), whether line-level tests strip first, in what order the nine rules
short-circuit, and whether boundary values pass — a one-bit difference in any of these yields a
different surviving corpus.

> To keep judging **deterministic**: tokenization is pinned to `text.split()` whitespace splitting
> (upstream uses the NLTK Treebank tokenizer, unavailable in the dependency-free eval environment);
> the punctuation set is pinned to the 63 characters listed in the starter (the upstream set is
> larger; this problem takes all of ASCII punctuation plus common full-width marks, ellipsis and
> dashes); all nine rules are always on.

## Rules

- Implement `solve(text, min_doc_words=50, max_doc_words=100000, min_avg_word_length=3,
  max_avg_word_length=10, max_symbol_word_ratio=0.1, max_bullet_lines_ratio=0.9,
  max_ellipsis_lines_ratio=0.3, max_non_alpha_words_ratio=0.8, min_stop_words=2)`, checking **in
  this order**, short-circuiting on the first failing rule (`failed_rule`); all pass →
  `passed=true`:

  1. `gopher_short_doc`: non-punctuation word count `< min_doc_words` fails (`=` passes)
  2. `gopher_long_doc`: non-punctuation word count `> max_doc_words` fails
  3. `gopher_avg_word_length`: the mean length (punctuation chars included) of non-punctuation
     words outside `[min, max]` fails
  4. `gopher_too_many_hashes`:
     `text.count("#") / n_words > max_symbol_word_ratio` fails — the denominator is the **total
     word count** (punctuation-only words included); using non-punctuation words there is a common
     implementation bug
  5. `gopher_too_many_ellipsis`:
     `(count("...") + count("…")) / n_words > max_symbol_word_ratio` fails
  6. `gopher_too_many_bullets`: share of lines starting (after `lstrip()`) with `•` or `-`
     `> 0.9` fails
  7. `gopher_too_many_end_ellipsis`: share of lines ending (after `rstrip()`) with `...` or `…`
     `> 0.3` fails
  8. `gopher_below_alpha_threshold`: share of words containing at least one alphabetic character
     (`isalpha`) `< 0.8` fails
  9. `gopher_enough_stop_words`: intersection size with the 8 stop words
     {the, be, to, of, and, that, have, with} `< 2` fails

- Return a dict: `passed`, `failed_rule` (the first failing rule's reason string or None),
  `checks` (the nine rule names → bool, in the order above), and the ledger fields
  `n_words` / `n_non_symbol_words` / `avg_word_length` / `bullet_lines_ratio` /
  `end_ellipsis_lines_ratio` / `alpha_words_ratio` / `n_stop_words`.

## Example

```
Input: text = "the model that we have is small and it works fine today" (12 words)
  min_doc_words=50 → 12 non-punctuation words < 50 → first rule short-circuits
Output: passed=false, failed_rule="gopher_short_doc";
        later checks are still computed but do not short-circuit
```

## Constraints

- Pure Python (torch not needed); empty text is valid input (`n_words=0`; ratio ledgers guard
  against division by zero)
- A word is a `text.split()` token; a **punctuation-only word** (every char in the punctuation set)
  is excluded from the non-punctuation count but included in `n_words` and ratio denominators
- A line is a `text.splitlines()` element; bullet tests use `lstrip()`, end-ellipsis tests use
  `rstrip()` (a `-` after leading whitespace is a bullet; `...` before trailing whitespace still
  counts)
- Threshold boundaries always use **strict inequality to fail**: exactly at the threshold passes
