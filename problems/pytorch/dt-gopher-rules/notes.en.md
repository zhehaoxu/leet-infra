# Solution Notes · Gopher Rule-Based Quality Filter

## Upstream source and determinization

Traced back to `huggingface/datatrove`'s
`src/datatrove/pipeline/filters/gopher_quality_filter.py` (the engineering version of the Gopher
paper, arXiv:2112.11446). Three deviations are required for judging, all pinned in the problem
statement:

1. **Tokenization**: upstream `split_into_words` uses the NLTK Treebank word tokenizer ("don't"
   splits into "do" + "n't"); the eval environment is dependency-free, so `text.split()` is pinned.
2. **Punctuation set**: upstream `PUNCTUATION_SET` is ASCII punctuation plus a long tail of unicode
   terminal punctuation (~400 chars); this problem pins a 63-char set (all ASCII punctuation plus
   `…—–·、，。！？；：""''《》【】（）〈〉「」『』`).
3. **Rules always on**: upstream allows disabling any rule by passing `None`; here all nine are
   always on and the ledger is fixed.

## Four error-prone details of the conventions

- **Two word counts**: document length (rules 1/2) uses the **non-punctuation word** count; the
  symbol-ratio denominator (rules 4/5) is the **total** word count. Mixing them breaks both rule 1
  and rule 4 ledgers at once.
- **avg's character convention**: the length of a non-punctuation word **includes punctuation
  characters inside it** ("don't" counts as 5 — if it survives as one split token). Upstream takes
  the mean of `len(w)` over non_symbol_words; we match that.
- **Direction of stripping**: bullets only `lstrip()` (a `-` at line end is not a bullet); end
  ellipsis only `rstrip()` (leading `...` does not count). Both `"..."` and `"…"` must be
  supported in two places (rule 5 counts characters, rule 7 tests line endings).
- **Short-circuit order**: the nine rules are checked in the stated order and the first failure
  returns its reason; but the `checks` ledger still computes all nine — a mutation that swaps two
  rules' order must be caught by the reason assertion.

## Testing and mutations

- The oracle (`_oracle`) is independently hand-written (no `import solution`), cross-validated on
  50 random texts (vocabulary includes punctuation words, ellipses, bullet markers, stop words)
  with randomly perturbed thresholds.
- Boundary anchors: non-punctuation count exactly 50/60 passes; all-10-letter words pass under
  `max_avg_word_length=10`; hash ratio exactly 0.1 (6 of 60 words) passes, 9 fails.
- One dedicated case per rule (built carefully so earlier rules do not short-circuit first: the
  hash/ellipsis/bullet cases must lay down 50+ non-punctuation words beforehand).
- All 9 mutations caught: short boundary loosened, long boundary tightened, avg made strict,
  hash denominator swapped to non-punctuation words, `…` dropped, bullet without strip, end
  ellipsis without strip, `any`→`all`, stop test `>=`→`>`. The endell-nostrip mutation initially
  had no observation surface (no trailing whitespace in test lines) and was caught after adding a
  `...` case with trailing spaces.
