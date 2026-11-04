# QUITO Query-Attention Filtering — Implementation Notes

## Source and conventions

- Official repo `Wenshansilvia/attention_compressor` (`quito/utils.py` + `quito/compressor.py`), paper arXiv:2408.00274 (QUITO, CCIR 2024). QUITO-X (the information-bottleneck view, EMNLP 2025 Findings) is out of scope.
- **Offline scoring**: the official `attention_score` depends on an HF forward pass (`apply_chat_template` + trigger token, a hard-coded `prefix=14` offset, last-layer attention averaged over heads with the **final row** taken, then `softmax(axis=0)`). The judge has only torch+numpy (no transformers), so attention scores are given directly and the graded part is the rule chain after scoring.
- **Official defect (critical!)**: `text_filter` calls `get_sorted_ids(score)`, but utils.py **never defines that function** (copying it verbatim raises NameError). Following the paper's "keep high-attention content", it is pinned as the **stable descending-score index**; the adjacent `get_rank_list` (ascending rank map) is its mirror reference. Do not look for an "official definition" — there is none.

## Per-entry details

### reconstruct
- A token **starting with** `Ġ` (GPT-2 byte-BPE space prefix) or `▁` (SentencePiece underline) starts a new word; otherwise it is appended to the current word, attention pooled by **max**, cnt += 1.
- The `Ċ` → `\n` replacement happens **before** the prefix test: `"ĊĠnew"` becomes `"\nĠnew"`, which does not start with `Ġ`, so it stays one token-word (newlines do not split).
- **Official quirk**: the first prefix-less token (e.g. "hel" in `["hel","lo","Ġworld"]`) goes through the else branch and increments cnt from its initial 1 → the first word gets cnt=3, not the intuitive 2. Pinned as-is.
- The last word is flushed after the loop.

### gaussian_smooth
- numpy reimplementation of `scipy.ndimage.gaussian_filter1d` (defaults `mode='reflect'`, `truncate=4.0`):
  - `radius = int(truncate*sigma + 0.5)` (σ=0.7 → radius 3; dropping the +0.5 gives 2, observable);
  - normalized kernel `exp(-x²/(2σ²))`; padding uses numpy's **`symmetric`** mode (scipy's `reflect` == numpy's `symmetric` — naming trap);
  - σ=0 returns identity.
- **Verification**: bit-exact against scipy for σ∈{0.5,0.7,1.0,1.5,2.0,3.0} × n∈{5,13,24,41} (max diff 5.6e-17). scipy was available only via system python3 (neither venv has it; installed with `--target /tmp/scipy_chk` into an isolated directory, no venv polluted).

### text_filter (word level)
- rank = **descending** stable index; `threshold = sum(cnt)*ratio` (**sum**, not len — with all cnt=1 they coincide, so a fixture with cnt>1 is required to distinguish).
- **Strict `<`**: a word whose accumulation lands exactly on the threshold is dropped and the loop breaks (bias toward keeping fewer). This is the **opposite** of sentence_filter's `>` — both boundary quirks are pinned deliberately.
- Scanning stops at the break (mathematically equivalent: acc is monotone, so the condition stays false afterwards — mutation m-tf-nobreak is therefore a true equivalence, removed and recorded; do not pin it again).
- Output preserves **original order** (not score order): with scores `[0.5, 0.9, 0.2]` and kept set {m, n}, the output is `[m, n]`, not `[n, m]`.

### sentence_filter (sentence level)
- The official code uses nltk `sent_tokenize` plus a fragile substring-matching loop; this problem takes sentence boundaries as an offline input `sizes` (words per sentence) and pins: sentence score = **max** of its word scores (not mean, not sum).
- `budget = len(words) * ratio` (float); greedy in descending score order, breaking only when `acc + size > budget` (**exact fits pass**), recording the breakpoint sentence and `budget_left = budget - acc`.
- Selected indices are `sort()`ed back into original order before joining.
- **When everything fits**, the official code falls into its `except` branch (breakpoint undefined) and returns None after printing — pinned here as `breakpoint_seg = budget_left = None`.
- Ties are stable: equal sentence scores keep original sentence order.

### dynamic_span (dynamic sentence level)
- `compress_sentence_token = sentence_filter + _select_tuples`: inside the breakpoint sentence that did not fit, start from its **highest-scoring word** (max takes the first, i.e. leftmost on ties) and expand two pointers **alternately left/right** (both sides at once when available) until `budget_left` words are collected or the sentence is exhausted (break); finally sort by original index.
- If m exceeds the sentence length, the whole sentence is taken.

## Verification chain

- Sandbox `/tmp/quito_sandbox.py`: four cross-check blocks (three official reconstruct quirks, text_filter strict-< + len/sum threshold + original order, sentence max + exact-fit pass + None breakpoint, dynamic expansion in four shapes) plus the all-combination scipy Gaussian check.
- selftest dual-mode 5/5 ACCEPTED; mut801 **23/23 all CAUGHT**, one true equivalence removed (m-tf-nobreak, reason above).

## Boundary with other ctx problems

- LLMLingua-2 (ctx-llmlingua2-word-compress) pins classifier-scored word pruning; LongLLMLingua pins perplexity-based budget allocation; GemFilter pins early-layer attention top-k input pruning.
- QUITO's increment: the whole post-scoring rule chain — **subword→word max aggregation with cnt accounting, Gaussian smoothing, strict-< cnt budget, sentence-level max greedy, and breakpoint-sentence centered expansion** — especially the opposite boundary conventions of the word and sentence levels.
