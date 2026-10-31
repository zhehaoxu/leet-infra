# notes — dt-c4-filter (order 664)

## Source

- Authoritative source: `huggingface/datasets` v1.2.1 `datasets/c4/c4_utils.py` (the TFDS c4 mirror; HF legacy-datasets/c4 shares it). Constants: `_MIN_WORDS_PER_LINE=5`, `_MIN_NUM_SENTENCES=3`, `_MAX_WORD_LENGTH=1000`, `_END_MARKS=(".","?","!","\"")`, `_ELLIPSIS="..."`, the 6 `_POLICY_SUBSTRINGS`. The line-filter order, the page-reject vs line-drop distinction, and the badwords regex `[^a-z]({})[^a-z]` are transcribed verbatim.
- Upstream counts sentences with NLTK punkt; the judge env has no NLTK, so the statement pins "1 sentence per valid line" as a deterministic proxy (deviation recorded here).
- Upstream dedup is a global Beam cogroup; this problem uses "within-page + explicit `seen_hashes` cross-page state", keyed by `md5(line.strip().lower())` (upstream `_hash_line` verbatim).

## Judging surface

- solve args: text, badwords (None = no filter), seen_hashes, three thresholds. Returns accepted/reject_reason/valid_lines/clean_text/num_sentences/drop_counters/new_hashes.
- Oracle: `test_cases._py` independently rewritten (step-wise replaces instead of one regex), all fields compared.
- Anchors: one hit for each of the six drop classes; citation stripping precedes endmark/word-count checks; the four page rejects (lorem/brace/badword/too-few-sentences); word-length 1000/1001 and word-count 4/5 boundaries; the badword anchor quirk (essex no-hit, ' sex ' hits); within-page dedup and cross-page seen state; dedup key strip+lower (case/whitespace variants collide).
- Invalid inputs: text type, empty strings in badwords, seen_hashes type, thresholds ≤0/bool/float → ValueError.

## Pitfalls (stepped on while authoring)

- Nearly every constructed test line gets intercepted by `_MIN_WORDS_PER_LINE=5` first (3–4-word lines never reach the javascript/policy checks) — count words before constructing.
- Page-reject semantics pinned: valid_lines cleared, num_sentences keeps the pre-rejection value, counters kept.
- The `hash-no-strip` mutation is equivalent (line is stripped at the loop head); the real mutation is `hash-case-sensitive`.

## Mutation log (9, all caught)

ellipsis-kept, longword-ge (boundary), minwords-le (boundary), javascript-rejects (line-drop → page-reject), lorem-startswith, badword-no-anchor, hash-case-sensitive, seen-not-updated, policy-first.

## Red lines

No NLTK/langdetect/Beam; no URL normalization, RealNews or WET parsing (boundary with `dt-gopher-rules`: that is the Gopher nine rules, this is C4 line filtering + dedup).
