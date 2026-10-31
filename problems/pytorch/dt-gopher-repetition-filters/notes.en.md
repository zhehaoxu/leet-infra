# notes — dt-gopher-repetition-filters (order 680)

## Source

- Authoritative source: the Gopher paper (Rae et al., arXiv 2112.11446), **Appendix A.1.1** Repetition Removal subsection and the **Table A1** threshold list (verified by verbatim PDF extraction). Key sentence: "For lines and paragraphs separately, we calculate over the document both the fraction that are duplicates, and the fraction of characters contained within those duplicates; for each n ∈ {2,…,4}, we calculate the fraction of characters contained within the most frequently-occurring n-gram; and for each n ∈ {5,…,10}, we calculate the fraction of characters contained within all duplicate n-grams, taking care not to count characters that occur in overlapping n-grams more than once."
- Table A1's thirteen entries: dup line 0.30 / dup para 0.30 / dup line char 0.20 / dup para char 0.20 / top2 0.20 / top3 0.18 / top4 0.16 / dup5 0.15 / dup6 0.14 / dup7 0.13 / dup8 0.12 / dup9 0.11 / dup10 0.10. The rejection word "surpasses" implies strict inequality.
- DeepMind never open-sourced this pipeline and the paper does not define lines/paragraphs/n-grams precisely; the following conventions are pinned in the problem statement (deviations recorded here): lines = splitlines verbatim (blank lines count as repeatable); paragraphs = maximal non-blank line runs separated by blank lines, paragraph characters = sum of line lengths; words = whitespace-split with character spans; n-grams are word-level exact matches (the paper's "normalize white spaces and ignore punctuation" is written only in the MinHash 13-gram paragraph, not the repetition one); top n-gram ties break by earliest first occurrence; coverage fractions divide by len(text).
- Boundary with `dt-minhash-lsh-dedup` (order 551): that problem is the inter-document MinHash 13-gram Jaccard ≥ 0.8 near-duplicate dedup (the paper's Document Deduplication paragraph); this problem is the intra-document Repetition Removal exact slicing ledger — adjacent in the paper, entirely different mechanisms.

## Judging surface

- solve(text, thresholds=None) → {accepted, reject_reason, metrics(13 names)}; thresholds overrides any subset with key validation and value range [0,1] (bool rejected).
- Oracle: test_cases._py independently rewritten — line/para duplication via double loops, n-gram coverage via a per-character boolean mask (a different path from the solution's interval merging), full-field comparison.
- Anchors: full 13-metric comparison; instances vs distinct (A×3 document); exact-equality non-rejection at 0.30 line fraction and 0.20 line char (thresh-ge boundary); the blank-lines-participate quirk (5 blanks + x×2 = 7/15); a construction where the paragraph metric is the reason (6 paragraphs, 27 lines, line metrics pinned below thresholds); para char dominated by line char (paragraph duplication ⇒ line duplication with a shared denominator, so the para char metric can never fire first); top ties by first occurrence (both all-count-1 and count-2 scenarios); overlapping span merging (periodic document dup5 = 1.0 vs naive accumulation 3.79); the Table A1 order ledger (first in row order among several exceeded metrics); threshold-override reclassification; empty document / whitespace-only / too few words.
- ValueError: text type, thresholds not a dict / unknown key / bool / string / out of range.

## Pitfalls (hit while authoring)

- Three consecutive hand-computed anchor errors: (1) paragraph duplication necessarily causes line duplication (paragraphs are made of lines) and the two char fractions share denominators, so dup line char ≥ dup para char — a "para char fires first" test is unconstructible; (2) blank lines are lines and blank-heavy documents nearly always trip the line fraction before the paragraph fraction (the para-reason case needed 27 lines across 6 paragraphs); (3) the periodic document's dup5 coverage is 1.0 (the last word's end clamps to len(text)=29), not the mis-computed 30/29.
- Word repetition ≠ line repetition: line matching is whole-line exact, so line-ledger fixtures must repeat entire lines.
- Line characters include intra-line whitespace; the coverage denominator len(text) includes newlines and inter-paragraph whitespace while coverage itself only counts word characters — two different conventions to keep straight.

## Mutation record (10/10 caught)

thresh-ge (>= boundary), dup-line-distinct, span-no-merge, top-tie-last (tie by latest), top-span-single (top counts only the first occurrence), para-always-split (zeroes the para ledger), line-char-denom-text, metric-order-reversed, dup-ngram-min-count (count>=3), skip-blank-lines. No equivalent mutations.

## Red lines

Gopher's quality heuristics (word counts / symbol ratios / bullet lines) belong to `dt-gopher-rules` (order 656); MinHash near-dedup belongs to `dt-minhash-lsh-dedup`; C4 line cleaning belongs to `dt-c4-filter`. Test-set filtering (13-gram Jaccard against test sets) is not modeled (the random one-of-two removal is non-deterministic). No torch dependency — a pure-Python ledger problem.
