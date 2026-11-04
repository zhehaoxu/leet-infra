# Problem

**CPC** (Context-aware Prompt Compression, arXiv:2409.01227, AAAI-25, Workday) trains a bidirectional-attention sentence encoder that scores every sentence by cosine similarity between the question vector and the sentence vector, then keeps or drops **whole sentences** against a token budget — avoiding the semantic damage of token-level deletion.

This problem reproduces the deterministic selection chain that comes after scoring. Encoder scoring (model forward + mean pooling + cos_sim), tokenizer length measurement, and regex hits are all offline inputs:

1. **boost_flags** — `re.match` semantics (**sentence start only** counts as a boost; a mid-sentence hit does not); pattern None → all False;
2. **rank_key** — a **two-key descending** stable index by `(boost, score)`: all boosted sentences first (descending score within the group), the rest by descending score, ties keep original order;
3. **select_segments** — the official greedy: `taken` starts at **-1** (compensating the +1 space charge), each sentence adds `taken += ln + 1`; the **overflowing sentence is taken before breaking** (the add-then-break quirk: the sentence that blows the budget is still included); an exact fit (`taken == target`) **continues**; fake (question echo) sentences are always skipped; the **output preserves original sentence order** (not score order).

## Task

Implement `solve(mode, **kw)` with three modes:

- `boost_flags(segments, pattern)` → a boolean list;
- `rank_key(boosts, scores)` → the descending stable index list;
- `select_segments(segments, scores, boosts, fakes, lengths, target)` → `(selected_segments, sorted_indices, taken)`.

Inputs are lists; outputs must match the anchors item by item.

## Constraints and conventions

- sentences ≤ 32; scores are finite floats; lengths are per-sentence token counts (≥ 1);
- `target` is an integer token budget; fake sentences participate in ranking but are skipped during selection;
- identical `(boost, score)` pairs always resolve stably (original order);
- when every sentence is fake, return an empty list and leave `taken` at -1.
