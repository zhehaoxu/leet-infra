# Problem

**QUITO** (Query-gUIded aTtention cOmpression, arXiv:2408.00274, CCIR 2024) runs a small model on a trigger question, takes the question's attention over the context tokens as a relevance score, and filters low-attention content against a budget — compression happens before the main model ever reads the context.

This problem reproduces every deterministic filtering rule that comes after scoring. The attention distribution (which depends on an HF forward pass: chat template, trigger token, last-layer attention averaged over heads with the final row softmaxed) is given as an offline input; you implement the rule chain from scores to compressed output, which contains no randomness:

1. **reconstruct** — subword aggregation: a `Ġ`/`▁` prefix starts a new word; continuation tokens are appended to the current word with **max** attention pooling and an incremented cnt;
2. **gaussian_smooth** — Gaussian smoothing of the attention sequence (scipy `gaussian_filter1d`, `mode='reflect'`, `truncate=4.0`; sigma=0 is identity);
3. **text_filter** — word-level: greedy in **descending** score order, keep while the accumulated cnt stays **strictly below** `sum(cnt)*ratio`, break immediately once it no longer fits, and emit results in **original order**;
4. **sentence_filter** — sentence-level: a sentence's score is the **max** of its word scores, the budget is `len(words)*ratio`, greedy in descending score order, break only when **strictly exceeding** the budget (exact fits pass), and selected sentences are sorted back into original order;
5. **dynamic_span** — dynamic sentence-level: inside the breakpoint sentence, expand **alternately left and right** from the highest-scoring word until the remaining budget is met, then restore original order.

## Task

Implement `solve(mode, **kw)` with five modes:

- `reconstruct(tokens, attention)` → `(words, attns, cnts)`;
- `gaussian_smooth(scores, sigma)` → the smoothed list;
- `text_filter(words, score, cnt, ratio)` → `(res_words, res_scores, keep)`;
- `sentence_filter(words, attention, sizes, ratio)` → `(selected_sentences, breakpoint_seg, budget_left)`;
- `dynamic_span(bp_words, bp_attn, m)` → `(selected_words, selected_attns)`.

Inputs are lists; outputs must match the anchors item by item (float tolerance 1e-9).

## Constraints and conventions

- tokens ≤ 64, words ≤ 32, sentences ≤ 8; scores are finite floats;
- `cnt` values produced by aggregation are always ≥ 1; `sizes` sum to the word count;
- when every sentence fits, the breakpoint segment and `budget_left` are `None`;
- ties always use **stable sorting** (equal scores keep original order), and max takes the first maximum.
