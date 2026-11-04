# CPC Sentence-Level Relevance Selection — Implementation Notes

## Source and conventions

- Official repo `Workday/cpc` (AAAI-25, arXiv:2409.01227). The core account lives in `util/preprocessing.py::compress_sample` (`prompt_compressor.py` is just a shell); the encoder is a trained bidirectional model (mistral/llama LoRA, `LlamaBiForMNTPandSentEmbeddingsV2`), made an offline input for the judge.
- **Offline input mapping**: sentence score = cos_sim(sentence embedding, mean_pooling(question embedding)); sentence token length = tiktoken/GPT-4 tokenizer; boost = `re.match(boost_sents_regexp, seg)`; fake = `segments_labels==1` (question-as-suffix echo sentences appended into the context). This problem takes `(segments, scores, boosts, fakes, lengths, target)` directly.
- **Budget semantics**: the official `compression_target_tokens` is a **token budget** (not a ratio); `taken` starts at **-1** as the official compensation for the per-sentence `+1` (space separator) — so the budget counts "tokens + separators".

## Per-entry details

### boost_flags
- `re.match` anchors at the **sentence start**: `"xEND tail"` does not match `END`, `"END head"` does. The official boost semantics = regex-matched sentences (e.g. instruction sentences) unconditionally outrank all ordinary ones.
- pattern None → all False (the official `re_checker` returns False when `boost_match_regex is None`).

### rank_key
- `sorted(all_similarities, reverse=True, key=lambda x: (x[3], x[1]))` — **reverse=True over the two-key tuple**: boost (True > False) first, then score. Python's `sorted(reverse=True)` is **stable**: full ties keep original order (do not replicate by negating scores — bool/negative pitfalls; just use reverse=True with the tuple key).

### select_segments
- Official verbatim: `taken = -1`; `taken += len(openai_tokenizer.encode(seg)) + 1`; `indices.add(idx)`; `if taken > target: break`.
- **add-then-break quirk**: the sentence that first pushes `taken` past target **is still included**, then selection stops — the compressed output may exceed the budget.
- **Exact fit continues**: when `taken == target`, `>` is false and selection continues (opposite of QUITO text_filter's strict `<`; same as its sentence_filter's `>`).
- **Fake skip**: fake sentences participate in ranking but are `continue`d during selection — with all fake, indices is empty and taken stays -1.
- **Original-order output**: `[seg for ... in all_similarities if idx in indices]` — filtered by the original global_index, not score order.

## True equivalence on record (do not pin again)

- **m-sl-breakfirst**: changing add-then-break into break-then-add (test `taken > target` before taking) leaves the final state **identical** — when the overflow sentence is taken, the previous taken is still ≤ target in both orders, so both take it and stop; the selected set and taken match exactly. Mutation ineffective, removed.

## Boundary with other ctx problems

- QUITO (=801): word/sentence-level **attention-score** filtering + Gaussian smoothing + dynamic centered expansion; CPC here uses **trained-encoder scores** + regex boost + whole-sentence token-budget greedy, no smoothing, no reordering.
- LLMLingua-2: classifier-scored **word-level** deletion; LongLLMLingua: perplexity budgets **allocated per document**; GemFilter: early-layer attention top-k. CPC's increment: boost priority, fake skipping, the `-1 start + ln+1` budget account, add-then-break.

## Verification chain

- Sandbox `/tmp/cpc_sandbox.py`, five cross-check blocks (boost/rank/select basics + fake/all-fake + exact-fit continue/overflow-included + re.match semantics + stable ties).
- selftest dual-mode 5/5 ACCEPTED; mut802 **11/11 all CAUGHT** + one true equivalence removed (above).
- New staging convention: while a guardrail runs, stage everything under /tmp and dry-run all verification (`mut802_dry` pointed at the staging dir); land into problems/ only after the guardrail finishes — avoiding the half-registered state that the full suite caught (the 320 errors of the 800-period run, task HpuENv).
