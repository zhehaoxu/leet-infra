# Solution Notes: Vocabulary Coverage and OOV Statistics

## Source and statistical conventions

This task uses engineering conventions for corpus statistics: report corpus occurrences, document frequency, token coverage, and type coverage separately. Filtering, exact matching, zero-denominator behavior, and OOV ordering are defined by this problem's contract. They do not depend on a specific external library or require a definition borrowed from a research paper.

## 1. Validate the vocabulary before filtering tokens

Build a mapping from each vocabulary token to its original index. If its size differs from the vocabulary length, raise `ValueError` for duplicates. An empty corpus or a duplicated token being ignored does not bypass validation.

Convert ignored tokens to a set for membership checks; repeated ignored entries have no statistical effect. For each document token, check whether it is ignored before updating any statistic. Do not fold case, normalize Unicode, match substrings, or modify input lists.

## 2. Separate occurrences from document frequency

Initialize two zero-filled arrays aligned with the vocabulary: corpus counts and document frequencies. Maintain OOV count dictionaries keyed by each original token.

For each document:

1. Every retained occurrence increments the retained document count; an OOV occurrence also increments its OOV count.
2. Increment the corresponding corpus occurrence count and add the token to this document's `seen` set.
3. After scanning the document, increment document frequency once for each token in `seen`.
4. Always append `(kept_count, oov_count)`, including `(0, 0)`.

Create a fresh `seen` set per document. A corpus-wide set would incorrectly cap document frequencies at 1. Conversely, incrementing document frequency on every occurrence would confuse it with occurrence counts.

## 3. Keep OOV outputs aligned

After processing all documents, sort OOV tokens by `(-count, token)`. Build `oov_counts` and `oov_doc_freq` using that same ordered token list; do not sort the three arrays independently.

Python lexicographic order compares the original strings' code points, without locale-aware collation. Case, numeric-looking strings, and precomposed versus combining forms remain distinct. Collapsing OOV tokens into an unknown ID loses identities needed for type counts, ordering, and document frequency.

## 4. Coverage denominators are different

Let $K$ be retained in-vocab occurrences, $O$ retained OOV occurrences, $V_+$ the number of vocabulary entries with positive counts, and $U$ the number of distinct retained OOV tokens:

$$
\mathrm{token\_coverage}=\begin{cases}K/(K+O),&K+O>0\\0.0,&K+O=0\end{cases}
$$

$$
\mathrm{type\_coverage}=\begin{cases}V_+/(V_++U),&V_++U>0\\0.0,&V_++U=0\end{cases}
$$

Neither denominator is the original unfiltered occurrence count or the vocabulary size. Unobserved vocabulary entries do not affect type coverage. Python division produces a `float`; explicitly return `0.0` for a zero denominator.

## Executed example walkthrough

- Example 1 has `vocab_counts=[0, 2, 0]` and `vocab_doc_freq=[0, 1, 0]`. OOV token `x` occurs twice across two documents. The document statistics are `[(3, 1), (0, 0), (1, 1)]`; both coverages are `0.5`.
- Example 2 ignores `É` but retains `é`; the combining form remains a separate OOV token. There are two known occurrences and OOV counts `[3, 1, 1]`. The actual floating-point token coverage is `0.2857142857142857`, while type coverage is `0.25`.

## Complexity

Let $T$ be input token occurrences, $V$ vocabulary length, $I$ ignored-list length, $D$ document count, and $U$ retained OOV types. Assuming amortized constant-time hashing and bounded-length string comparisons, time is $O(T+V+I+D+U\log(U+1))$ and space including returned data is $O(V+I+U+D)$. A document's `seen` set cannot exceed the total number of retained known and OOV types. Empty inputs additionally incur constant overhead.

## Independent verification

The 15 `check_*` functions cover the two denominators, repeated occurrences versus document frequency, ignored tokens inside and outside the vocabulary, empty corpora/documents, all-ignored inputs, unused vocab entries, all-known/all-OOV inputs, frequency ordering and Unicode lexical ties, invalid duplicate vocabulary entries, and immutability. Boundary cases include 30 documents of 100 tokens each, 100 vocabulary entries, and 30-character tokens. The 180 randomized cases use seed 8202.

The oracle neither uses `Counter` nor repeats the single-pass accumulation approach. It first builds retained documents and a flattened corpus, then scans that corpus separately for each vocabulary entry and observed OOV type. Document frequencies use membership checks per document. Coverage is independently calculated from the flattened corpus and observed set. Tests additionally enforce exactly eight keys, Python integer/float types, lists and document-stat tuples, and preservation of nested inputs.
