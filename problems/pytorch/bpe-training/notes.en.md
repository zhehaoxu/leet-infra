# BPE Training Explained

## Why BPE

A vocabulary that is too large (whole words) explodes with OOV, while one that is too small (single characters) makes sequences too long. BPE is the compromise: starting from a character vocabulary, it **greedily merges the most frequently co-occurring adjacent pair into a new symbol**, repeating $K$ times to obtain $K$ subwords. Frequent words (such as `the`, `ing`) become a single token, while rare words stay fragmented — vocabulary size is controlled and common patterns are naturally favored. GPT-2 uses 50k-level merges, and LLaMA/Qwen follow the same family (some use BPE variants such as "unigram/WordPiece", which share the idea).

## The three-step loop

```
count adjacent pairs (frequency-weighted) → pick the largest pair (ties: lexicographically smallest) → non-overlapping replacement in all words
```

Key details, all of them point-losers:

1. **Frequency weighting**: the count is not "how many words contain this pair" but the sum over words weighted by frequency. `e r` in `newer:6` adds +6 directly, not +1.
2. **Ties go to the lexicographically smallest**: when several pairs have the same count, `min(pairs, key=lambda p: (-cnt[p], p))`. `p` is a bigram, so the first symbol is compared before the second. **Without pinning this down, training results are not unique**, and judging breaks.
3. **Non-overlapping, left to right**: merging `('a','a')` in `a a a` yields `aa a` (consume two slots, then skip), not three pairwise merges. Use a greedy scan with `i+=2`.
4. **`</w>` is an ordinary symbol**: `w` and `</w>` may merge into `w</w>`, so word endings get folded into subwords (as in `er</w>` in the example). After merging, the symbol is a string concatenation and may be multi-character.

## Training / encoding are a pair

This problem only produces the **rule sequence**; the downstream `bpe-encoding` takes the same sequence and, for unseen words, "applies the earliest available pair according to the rule **priority** (learning order = rank)". So the **order** returned by training is itself information — do not sort it and do not deduplicate it.

## Complexity note (not required)

Naively rescanning the whole corpus every round is $O(K\cdot|\text{corpus}|)$. Teaching-sized inputs do not need optimization; real implementations maintain an inverted index from pair → word set and update counts incrementally, avoiding a full scan each round.

## Related

- Downstream: `bpe-encoding` (encode using the merges learned here).
- Comparison: `one-hot-encoding`/`embedding-lookup` in operators are the stages after tokenization.
- Judging: pure strings/dicts, no randomness, no torch; `check_example` uses dumped values from the classic low/newer/wider corpus, plus a dedicated tie-break test case.
