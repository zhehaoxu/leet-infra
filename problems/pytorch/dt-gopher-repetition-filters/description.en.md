# Gopher Repetition Filtering: Line/Para/n-gram Thirteen-Threshold Ledger

## Problem

Gopher (Rae et al., 2112.11446), Appendix A.1.1, MassiveWeb **Repetition Removal**: compute 13 repetition metrics on a document (Table A1) and reject the whole document when any metric **strictly exceeds** its threshold ("surpasses"). This problem transcribes that intra-document repetition ledger — in contrast to `dt-minhash-lsh-dedup`, which is the inter-document MinHash 13-gram Jaccard near-duplicate dedup; here it is exact n-gram slicing and counting.

## Solution

The 13 metrics and pinned conventions (all exact matching, case- and punctuation-sensitive):

1. **Lines**: every line of `text.splitlines()` verbatim (no strip; **blank lines participate in repetition**).
   - `duplicate_line_fraction` = duplicate line instances / total lines (a line appearing k≥2 times contributes k instances);
   - `duplicate_line_character_fraction` = characters in duplicate line instances / total line characters.
2. **Paragraphs**: maximal runs of non-blank lines separated by blank lines (strip-empty); paragraph characters = sum of its line lengths (excluding separators and intra-paragraph newlines).
   - `duplicate_paragraph_fraction` / `duplicate_paragraph_character_fraction` follow the same instance convention.
3. **Words**: maximal non-whitespace runs (with character spans [s, e)); a word n-gram = n consecutive words, and an occurrence covers [first word s, last word e).
   - `top_n_gram_character_fraction` (n=2,3,4): the most frequent n-gram (ties broken by **earliest first occurrence**), union of its occurrence spans / len(text);
   - `duplicate_n_gram_character_fraction` (n=5..10): union of the occurrence spans of **all** n-grams with count ≥2 / len(text) — overlapping spans are merged so characters are not double-counted (per the paper's "taking care not to count characters that occur in overlapping n-grams more than once").

Zero denominators, len(text)==0, or fewer than n words yield 0.0; every coverage fraction is divided by len(text) (whitespace and newlines included) while coverage itself only counts word characters.

**Rejection**: scan in Table A1 row order (line → paragraph → line char → para char → top2 → top3 → top4 → dup5..dup10) and report the first metric with `metric > threshold` as `reject_reason`; the `thresholds` argument may override any subset (keys must be valid metric names, values in [0,1]).

## Example

```
Input:  text = "x\n\nx\n\na\nb\n\nc\nd\n\ne\nf\n\ng\nh"
Output: reject_reason = "duplicate_line_fraction" (5 blank lines + "x"×2 = 7 duplicate instances / 15 lines)
```

## Constraints

- text is a str; thresholds is None or a valid dict, otherwise raise `ValueError`.
- Return `{"accepted", "reject_reason", "metrics"}` with all 13 metric names mapped to floats.
