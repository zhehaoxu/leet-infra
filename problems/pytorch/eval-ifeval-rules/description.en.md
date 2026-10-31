# IF-Eval Verifiable Instructions: Rule-Judge Ledger

## Problem

IF-Eval (google-research/instruction-following-eval) evaluates instruction following with **programmatically verifiable instructions**: keyword existence/forbidden/frequency, word-count constraints, end phrase, bullet-list counts, etc. This problem transcribes 6 judge families in the strict convention: given `instruction_ids + kwargs_list + response`, emit per-instruction `observed` (the measured count) and `pass`, plus the aggregate `all_pass`.

## Solution

Conventions are pinned by the statement (NLTK proxies replaced by explicit rules):

- `keywords:existence`: substring, case-insensitive, all must hit;
- `keywords:forbidden`: same substring convention, hit count must be 0;
- `keywords:frequency`: matches of `r"\b<kw>\b"` on the lowered response (word boundaries keep `spa` from hitting `spam`), compared per `at least`/`less than`;
- `length_constraints:number_words`: `len(response.split())` (whitespace split, punctuation stays attached);
- `format:end_phrase`: endswith after `response.strip().rstrip('"').lower()` (trailing quotes stripped, case-insensitive);
- `detectable_format:number_bullet_lists`: lines starting with `* ` ≥ num_bullets (`- ` does not count).

## Data scale

- 1 ≤ len(instruction_ids) ≤ 8; ids must be in the 6-family whitelist.
- Length mismatch, unknown id, empty keywords, frequency/num_words ≤0 or bool, bad relation, empty end_phrase, non-str response → `ValueError`.
