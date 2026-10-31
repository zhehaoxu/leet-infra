# Context Compression: LLMLingua-2 Word-Level Keep-Probability Merge and Threshold

## Problem

LLMLingua-2 (2403.12968) uses a BERT-size encoder to give each token a binary label, takes the "keep" class probability after `softmax`, then merges probabilities per word and thresholds against the target compression rate, dropping low-probability content. What actually decides the compressed output is the post-processing: which subwords belong to the same word, how force tokens are pinned to 1.0, how consecutive duplicate forced words are handled, and whether the threshold is taken over words or tokens. This problem loads no model — the token strings and per-token keep probabilities are given, and you reproduce the post-processing.

## Task

Implement `solve(tokens, token_probs, reduce_rate=0.5, token_to_word="mean", force_tokens=None, token_map=None, force_reserve_digit=False, drop_consecutive=False, special_tokens=None, model_name="xlm-roberta-large", token_count=None)`:

- `tokens`: `list[str]`, one string per token;
- `token_probs`: `list[float]`, same length as `tokens`, keep probability per token in `[0, 1]`;
- `reduce_rate`: `0 < rate < 1`, target deletion ratio;
- `token_to_word`: `"mean"` (average subword probabilities) or `"first"` (take the first subword);
- `force_tokens`: words kept regardless of probability (matched by pure token form);
- `token_map`: `{original: replacement}` — the replacement token is mapped back to the original word during merging;
- `force_reserve_digit`: when true, tokens containing digits get probability `1.0`;
- `drop_consecutive`: when true, only the first of consecutive duplicate forced words is kept;
- `special_tokens`: skipped entirely, never enter the word list;
- `model_name`: only `"xlm-roberta-large"` (`▁` starts a new word) and `"bert-base-multilingual-cased"` (`##` continues) are supported;
- `token_count`: callable `word -> int`, the number of tokens this word produces in the target tokenizer; `None` means 1.

Return a `dict` with keys `words`, `word_probs`, `keep_words`, `keep_labels`, `threshold`.

## Rules

1. Tokens in `special_tokens` are skipped.
2. Begin-of-word test: under `xlm-roberta-large`, a token that is punctuation, in `force_tokens` / `token_map.values()`, or starts with `▁`; under `bert-base-multilingual-cased`, `##`-stripped form hits force/map, or the token does not start with `##`.
3. A begin-of-word token whose pure form hits `force_tokens` or `token_map.values()` gets probability `1.0`; afterwards `token_map` replacements are mapped back to the original word.
4. A continuation token has its `##` / `▁` stripped and is appended to the previous word; if `force_reserve_digit` and the token contains a digit, its probability becomes `1.0`.
5. Word probability merges subword probabilities by `token_to_word` (`mean` or `first`).
6. With `drop_consecutive=True`, first take `percentile(word_probs, int(100*rate))` as the between-words threshold; a forced word resets the state if a word above that threshold appeared since the last forced word, otherwise a duplicate of the previous forced word gets probability `0`.
7. Expand every word probability `token_count(word)` times into a token-level list and take `threshold = percentile(expanded, int(100*rate + 1))` (linear interpolation, `+1` then truncate).
8. Keep `p > threshold`, or also `p == threshold` when `threshold == 1.0`; under `drop_consecutive`, a forced word equal to the last kept word gets label `0`.

## Example

```
tokens = ["▁Hello", "▁world", "▁Hello", "▁world"]
token_probs = [0.9, 0.4, 0.8, 0.3]
reduce_rate = 0.5

words       = ["▁Hello", "▁world", "▁Hello", "▁world"]
word_probs  = [0.9, 0.4, 0.8, 0.3]
threshold   = 0.612
keep_words  = ["▁Hello", "▁Hello"]
keep_labels = [1, 0, 1, 0]
```

## Constraints

- `tokens` and `token_probs` are non-empty and of equal length; probabilities are real numbers in `[0, 1]`.
- The first non-special token must begin a word, otherwise raise `ValueError`.
- `reduce_rate` must lie in `(0, 1)`; `model_name` and `token_to_word` accept only the listed values.
- `token_count` must return an `int >= 1`; `None` counts every word as 1 token.
- Input lists must not be modified; all invalid inputs raise `ValueError`.
