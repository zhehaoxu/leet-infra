# BPE Encoding Explained

## What training produces is a "priority", not just a "set"

The merges returned by `bpe-training` are **ordered**; the index is each merge's priority. When encoding you cannot think in terms of a set and "merge all these pairs" — order affects the result, and you must **re-select the smallest-rank (earliest) currently available pair each round** and apply it first. The reason: earlier merges **create** new symbols and new mergeable pairs (e.g. only after merging `e r → er` does `er </w>` appear).

Correct loop:

```
while an adjacent pair matching merges exists:
    pick the pair with the smallest rank → replace non-overlapping occurrences in the whole word
```

## Difference from the naive "scan merges in order once"

Some people write it wrongly like this: for each rule in merges, scan the word and apply it. For **some** words the results are identical, but they differ when "a later merge re-creates an earlier mergeable pair" — because after one pass, pairs with earlier ranks may reappear due to later merges. Standard BPE (and GPT-2's `bpe` implementation) always "take the globally smallest-rank available pair each round", and this problem fixes that behavior.

## Non-overlapping, left to right

Replacing `('a','a')` in `a a a` yields `aa a` (consume two slots then skip), exactly the same as `_merge_pair` on the training side. Both ends must use the same merge logic, otherwise what training learns cannot be used at encoding time.

## Edge cases

- `merges=[]`: return `list(word) + ['</w>']` directly (pure character level).
- Unseen words (not in the training corpus) can still be encoded — the value of BPE: no OOV, degrading at worst to characters.
- The return value keeps `</w>` (real implementations eventually drop it or map it to a special token id; this problem keeps it to verify the segmentation visually).

## Related

- Upstream: `bpe-training` (produces the merges used here; the two problems share the same classic example data and can cross-check each other).
- Downstream: `one-hot-encoding`/`embedding-lookup` (subword → id → vector).
- Judging: pure strings, no randomness; `check_example` uses the dumped subword results, plus cases with empty merges / unseen words.
