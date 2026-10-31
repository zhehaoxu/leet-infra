# Reference Solution: Begin-of-Word Merging, Force-to-One and the "+1 Truncated" Threshold

## Implementation

1. Validate types and ranges first (equal lengths, probabilities in `[0,1]`, `rate ∈ (0,1)`, enum whitelists, `token_count` callable returning an `int >= 1`).
2. **Begin-of-word merging** (upstream `llmlingua/prompt_compressor.py::__merge_token_to_word`, L2255-2292):
   - tokens in `special_tokens` are `continue`d;
   - `is_begin_of_new_word` (`llmlingua/utils.py` L81-101): BERT-family — `##`-stripped form hitting force/map makes it a word start, otherwise any non-`##` token starts a word; xlm-roberta — punctuation or force/map membership makes it a word start, otherwise a `▁` prefix does;
   - a begin-of-word token takes `get_pure_token` (strip `##` / `▁`); if the pure form hits `force_tokens` or `token_map.values()`, its probability is pinned to `1.0`; then `replace_added_token` maps replacements back to original words, and the `force_reserve_digit` digit test applies to the **restored word**;
   - a continuation token appends its pure form to the current word and pushes its probability onto that word's subword list, with the digit test applied to the **raw token**;
   - upstream raises `IndexError` when the first non-special token is a continuation; this problem raises an explicit `ValueError` instead.
3. **Word probability merge** (`__token_prob_to_word_prob`, L2294-2302): `mean` averages the subword list, `first` takes its head; other modes raise `NotImplementedError` upstream and `ValueError` here.
4. **drop_consecutive** (inside `__compress`, L2385-2397): word-level threshold `percentile(word_probs, int(100*rate))`; when scanning hits a forced word, an already-set `is_token_between` flag is merely reset, otherwise a duplicate of the previous forced word gets probability `0`; non-forced words accumulate the flag with `|=`.
5. **Expansion and threshold** (L2399-2409): every word probability is repeated `token_count` times (upstream `len(self.oai_tokenizer.encode(word))`, made an explicit `token_count` callable here); threshold is `percentile(expanded, int(100*rate + 1))` — the **`+1` and int truncation** are the two characters easiest to drop.
6. **Keep decision** (L2414-2429): `p > threshold`, or `p == threshold` when `threshold == 1.0` (protects all-ones input from keeping nothing); under `drop_consecutive` a forced word equal to the last kept word gets label `0` and is excluded from `keep_words`.

## Correctness Checks

9 `check_*` functions cover: the worked example (alternating `▁Hello/▁world`, threshold 0.612), a 10-case multi-knob oracle comparison (mean/first × both tokenizers × force/map/digit/special tokens/token_count/drop_consecutive), force and map pinning to 1.0, the mean/first difference on BERT `##` merging, zeroing consecutive forced words, special-token skipping, threshold/label changes under token expansion, threshold-equality keep at 1.0, and 21 invalid inputs.

The oracle (`test_cases.py::_oracle`) is an independent reimplementation sharing the contract, not the code.

Mutation testing (6/6 caught): mean→first (`bert_continuation_merge`/`matches_oracle`), force not pinned, threshold without `+1` (caught by 6 cases, example and oracle first), `>` → `>=`, drop_consecutive not zeroing, digit not forced.

## Scope and Sources

Upstream microsoft/LLMLingua (2403.12968):

- `prompt_compressor.py::__merge_token_to_word`: special skip, begin-of-word test, force/map pure-form pinning to 1.0, `replace_added_token` restoration, digit branches (restored token for word starts, raw token for continuations).
- `prompt_compressor.py::__token_prob_to_word_prob`: mean/first only.
- `prompt_compressor.py::__compress`: between-words threshold `int(100*rate)`, expansion via `oai_tokenizer.encode`, final threshold `int(100*rate + 1)`, keep condition `p > thr or (thr==1.0 and p==thr)`, consecutive forced-word label 0.
- `utils.py::is_begin_of_new_word / get_pure_token / replace_added_token`: the two tokenizers' rules (L81-120).

Deliberate contract differences:

- **No model or tokenizer is loaded**: encoder keep probabilities arrive as `token_probs`; tiktoken's `oai_tokenizer.encode` becomes the explicit `token_count` callable.
- Output `words` keep the `▁`/`##` prefixes (upstream `lstrip("▁")` for xlm happens after `keep_str` and label decisions, so compression decisions are unaffected; keeping prefixes makes the force-matching contract self-consistent here).
- Upstream has a keep-everything branch for `reduce_rate <= 0`; this problem restricts `rate` to the open interval `(0,1)`.
- Upstream chunks input by `max_seq_len` and compresses per chunk; this problem's input is one chunk.
