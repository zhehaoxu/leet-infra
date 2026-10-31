# C4 Line-Level Cleaning and Dedup: clean_page Filter-Order Ledger

## Problem

C4's (T5 corpus) core cleaning is the **line-level filter-order ledger** of `clean_page` (the HF datasets legacy `c4_utils.py` is the authoritative source). This problem transcribes that order line by line and attaches C4's line-level md5 dedup (a single-page + explicit-cross-page-state version of `remove_duplicate_text`): given a page's text with optional badwords and previously seen line hashes, return the kept lines, the cleaned text, per-filter drop counters (the deterministic counterpart of upstream's Beam counters) and the page-level rejection reason.

## Solution

Per line, after `strip()`, in upstream order:

1. any word longer than `max_word_length` (1000) → drop line;
2. strip citations `\[\d*\]|\[edit\]|\[citation needed\]`;
3. not ending in `.` `?` `!` `"`, or ending in `...` → drop line;
4. fewer than `min_words_per_line` (5) words → drop line;
5. `lorem ipsum` → **reject page**; `javascript` → drop line; `{` → **reject page**;
6. policy substrings (terms of use / privacy policy / cookie policy / uses cookies / use of cookies / use cookies) → drop line;
7. badwords regex `[^a-z](bw1|bw2|…)[^a-z]` hit → **reject page**;
8. dedup: `md5(line.strip().lower())` already seen → drop line, else keep and add to seen;
9. each valid line counts as 1 sentence (a deterministic proxy for upstream's NLTK punkt, pinned by the statement); total < `min_num_sentences` (3) → **reject page**.

Key points: the badword anchored regex does **not** match at word start/end or inside words (e.g. `sex` inside `essex` does not fire — an upstream quirk kept as-is); the dedup key is md5 of strip+lower (case/whitespace variants collide); on page rejection the drop counters so far are kept, and `num_sentences` counts valid lines found before the rejection.

## Data scale

- Pages of up to 64 lines; thresholds tunable but must be positive ints.
- Non-string text, empty strings in badwords, wrong seen_hashes type, thresholds ≤0/bool/float → `ValueError`.
