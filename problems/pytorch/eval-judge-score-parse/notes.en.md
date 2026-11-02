# notes (eval-judge-score-parse)

## Source

- Primary: `lm-sys/FastChat`, `fastchat/llm_judge/common.py` (main branch, fetched and checked)
  - Lines 34-37: the four regexes (pairwise primary / backup, single primary / backup), copied verbatim; `TIE_DELTA = 0.1` on line 30.
  - `run_judge_single` (135-189): `re.search(one_score_pattern, judgment)`, falling back to `_backup`, then `rating = -1`; on success it runs `ast.literal_eval(match.groups()[0])`.
  - `run_judge_pair` (235-310): the pairwise branch on lines 291-304 - primary pattern then backup, `scores = [ast.literal_eval(s.strip()) for s in match.groups()]`, `abs(scores[0]-scores[1]) <= TIE_DELTA` gives `"tie"`, otherwise the comparison yields `"A"` / `"B"`, and a miss yields `"error"`.
- Single-score table: `display_result_single` in `fastchat/llm_judge/show_result.py` (lines 9-36): line 20 filters `df[df["score"] != -1]`; lines 26-36 average over `(model, turn)` and `model` and print in descending order.

## Boundary

- `eval-judge-swap-consistency` (order 684) already owns **marker-style** verdicts (`[[A]]` / `[[B]]` / `[[C]]`) plus swap-two-game position mapping, disagreement-as-tie and half-win aggregation. That problem never touches numeric score strings, and this problem never touches swapping or win-rate tables - different input shapes, different parsed objects, different outputs.
- The first draft of this problem also implemented the swap mapping and `win_rate_adjusted`, overlapping 684 by roughly 60%. Under the "collision / no new mechanism, do not file" rule it was cut and rewritten to keep only numeric parsing plus the single-score mean table, which is exactly what the original todo entry asked for ("single-mode TIE_DELTA and the `[[a,b]]` dual-score regex").
- `tie_delta` is used here only for the A/B comparison **inside one text**, never for cross-text consistency or match settlement.

## Normalizations in this problem

- Upstream splits parsing across two files (`common.py` extracts, `show_result.py` averages) with pandas in between; this problem fuses them into one pure function, with the judge text as an explicit input and no model call at all.
- **The tie-break of the ranking must be pinned here**: `df.sort_values(...)` defaults to `quicksort`, which is not stable, so equal averages order by pandas internals and may shift across versions. This problem fixes it to "mean descending, then model name ascending".
- Parsing details were all measured (`.venv-eval`, CPython 3.13):
  - `\s?` is exactly one: `[[8,\t7]]`, `[[8,\n7]]`, `[[8,\xa07]]`, `[[8,\u30007]]` all hit; `[[8,  7]]` misses both tiers (`\s` eats one space and the spare space breaks the number slot).
  - `[[8,7],[9,2]]` misses the primary pattern entirely (the outer brackets are not `]]`) and the backup pattern returns the **leftmost** inner `[8,7]`; `[[8,7]] and [[9,2]]` takes the leftmost complete block via the primary pattern; `[[8, 7]]x` hits, `[[8, 7]x` falls back to the backup pattern.
  - `[[ 8, 7]]` misses both tiers (a digit must follow `[[`); `[[-1, 7]]` extracts nothing either (numbers carry no sign).
  - `5.` becomes `5.0` (float), while `007` makes `ast.literal_eval` raise **SyntaxError** (upstream leaves it uncaught and the server dies); this problem normalizes it to `ValueError`, and `check_rejections` guards that.
  - Three float boundary cases: `abs(8.0 - 8.1) = 0.09999999999999964 <= 0.1` gives a tie; `abs(1.0 - 1.1) = 0.10000000000000009 > 0.1` does not; and `abs(5.5 - 5.55) = 0.04999999999999982` falls **below** `tie_delta=0.05` even though the nominal gap equals it. Together they pin both sides of the `<=` float boundary.
- `s.strip()` is dead code upstream (a capture group holds only `\d` and `.`, never whitespace); this problem keeps the `int` / `float` distinction that `literal_eval` produces and asserts it in `check_number_kinds` / `check_contract`.
- `-1` is the "nothing captured" sentinel while `0` is a legal score: a dedicated `[[0]]` case exists so that idioms like `sc or -1` cannot swallow a zero (mutant `m-score-or`).

## Mutations

All 23 are CAUGHT (`.muts-judge2.json`, deleted after the run):

- Parsing: two `\s?` to `\s*` mutants (pairwise primary and backup), pairwise backup tried first, single backup tried first;
- Numbers: everything returned as `float`, `except` no longer catching `SyntaxError`;
- Verdicts: `<=` to `<`, A/B inverted, a miss turned into `"tie"`;
- Single table: `-1` kept, `avg` taken from `turn_1` only, `turn_1` / `turn_2` averaged while empty (division by zero), the turn ignored when bucketing (everything into turn 1), `sc or -1` swallowing a zero score;
- Ranking: ascending order, name tie-break removed;
- Validation: `turn` in `{0,1,2}`, `turn=True` allowed, `tie_delta=True` allowed, negative `tie_delta` allowed, `mode` check disabled, `entries` accepting only `list` (tuple rejected).

**Equivalent mutant (1, proven)**: `sc[0] > sc[1]` to `sc[0] >= sc[1]`. It looks like it could flip a tie, but it is unobservable - equal operands are always intercepted earlier by `abs(a-b) <= tie_delta` (with `tie_delta >= 0` covering zero), so the comparison never sees equal values; the semantics are identical and it is not listed.

Two mistakes worth recording from the first round: (1) `[[0008]]` placed in pairwise mode never reaches the parser at all (a single-number text), so leading-zero cases must use `[[007, 7]]` / `[[8, 08]]` where a token really is captured; (2) the first `m-pair-text-check` mutant was a no-op (`text = text`) and wasted a round - read what you change, even in a mutant.

## Judge notes

- Tolerance: means compare at 1e-12 (values like `22/3` must absorb 1e-16 noise); `scores` and `turn` stay exactly equal.
- The `pattern` label is part of the contract (`two` / `two_backup` / `one` / `one_backup` / `none`); asserting only on scores would throw away which tier matched.
