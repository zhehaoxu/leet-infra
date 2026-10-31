# notes — eval-ifeval-rules (order 665)

## Source

- Origin: `google-research/instruction-following-eval` (the REGISTRY in `instructions.py`; paper 2508.09108 "Instruction-Following Evaluation for Large Language Models"). Upstream judges use NLTK word_tokenize/sent_tokenize and have strict/loose conventions; the judge env has no NLTK, so the statement covers only the **strict convention** with pinned lexical proxies (word count = `split()`, frequency = `\b` regex).
- Only 6 families covered: keywords:existence / forbidden / frequency, length_constraints:number_words, format:end_phrase, detectable_format:number_bullet_lists. Upstream's remaining families (json_format, postscript, capital_word_frequency, combination:two_responses, sentence counting, etc.) are left as future candidates.

## Judging surface

- solve args: instruction_ids/kwargs_list/response. Returns results (per-instruction observed + pass) and all_pass.
- Oracle: `test_cases._py` independently rewritten (slicing instead of startswith, while-loop quote stripping, etc.), 120 random groups compared.
- Anchors: existence substring case-insensitivity; forbidden substring convention (`secrets` hits `secret`); frequency `\b` boundaries (`spa` misses `spam`, trailing punctuation does not block); word-count boundary 5 (both relations); end_phrase quote stripping + case-insensitivity + mid-text occurrence rejected; `* ` bullet counting with `- ` excluded; a 3-instruction combined all_pass.
- Invalid inputs: empty id list, unknown id, missing kwargs keys/empty lists, frequency/num_words/num_bullets ≤0 or bool, bad relation, empty end_phrase, non-str response → ValueError.

## Mutation log (9, all caught)

existence-or, forbidden-inverted, freq-rel-flip, words-count-chars, endphrase-case-sensitive, endphrase-no-quote-strip, bullets-dash-too, existence-case-sensitive, freq-no-boundary.

## Red lines

No model calls, no loose-convention markdown-stripping family; boundary with `rs-*` (reasoning systems): this is **instruction-following evaluation**, no candidate-aggregation semantics.
