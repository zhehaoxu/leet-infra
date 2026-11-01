# notes — eval-judge-swap-consistency (order 684)

## Source

- `lm-sys/FastChat` `fastchat/llm_judge/common.py` (run_judge_pair / play_a_match_pair) + `show_result.py` (display_result_pairwise), via GitHub API contents + base64.
- Official ledger points verified line by line:
  - **Parsing**: `if "[[A]]" in judgment: A elif "[[B]]": B elif "[[C]]": tie else: error` — **mark priority** ([[A]] always checked first), independent of the mark's position in the text; case-sensitive.
  - **Two swapped games**: `g1_map = {"A": "model_1", "B": "model_2"}`, `g2_map = {"A": "model_2", "B": "model_1"}`; `g1_map.get(w, w)` lets tie/error pass through unchanged.
  - **Aggregation condition** (show_result): `if row["g1_winner"] == "tie" or row["g1_winner"] != row["g2_winner"]: tie for both` — a tie in g1 or disagreement between the two games both count as a tie. Position bias (the judge still picking position A after the swap) lands exactly in "the two games each pick a model" → tie.
  - **Error row-level drop**: `df_all[(g1_winner != "error") & (g2_winner != "error")]` — errors are dropped before the tie rule, hence "g1 tie + g2 error → whole match dropped".
  - **Win rate**: `win_rate_adjusted = (win + 0.5*tie)/(win+loss+tie)`; comment verbatim: "each tie counts as 0.5 win + 0.5 loss".
  - The reference stores each match as one row with `g1_winner`/`g2_winner` twin fields (single mode fills both with the same value) — the problem keeps this output shape.
- MT-Bench paper (arXiv 2306.05685) §position bias: GPT-4 judgments show significant position bias; swap consistency is the countermeasure. The paper also reports an agreement metric (compute_agreement.py) — not tested here.

## Abstractions

- No real judge LLM call: g1/g2 judgment texts are given directly; the tested ledger is parse → map → resolve → aggregate.
- Single mode (one_score + TIE_DELTA=0.1) and the two-score format ([[a,b]] + regex backup) are **not tested** — recorded as follow-up candidates in notes.
- stats covering the full `models` list (absent models get all-zero rows) is a problem-side pin, not official behavior (the official DataFrame only contains models that appear) — pinned in the description to avoid ambiguity.

## Ledger points (nailed by tests)

1. **Consistent games = real win** (check_consistent_win): g1 A→alpha, g2 B→alpha (B holds answer_1) → alpha twice.
2. **Position-bias disagreement → tie** (check_position_bias_disagreement): still [[A]] after the swap → the two games pick different models.
3. **A tie in g1 ties the match** (check_any_tie_is_tie); g2's win/loss is discarded.
4. **Errors drop the match** (check_error_dropped; caught by error-as-tie).
5. **Error beats tie** (check_error_beats_tie): g1 tie + g2 error → dropped, not tied.
6. **Mark-priority quirk** (check_mark_priority_quirk; caught by b-first): with two marks in one text, the earlier-checked one wins.
7. **Case sensitivity** (check_case_sensitive_and_c_tie; caught by case-insensitive); [[C]] = tie (caught by c-as-b).
8. **Multi-match aggregation + half-win rate** (check_aggregation_over_matches; caught by adj-no-half / denom-short).
9. **Absent models all-zero rows, zero denominator → 0.0** (check_both_error_and_absent_models; caught by absent-skip / half-zero-to-one).
10. **Negative-direction swap** (check_loss_from_swapped_positions): g1 B→model_2 and g2 A→model_2 → model_1 gets a loss.

## Mutation record

12/12 all CAUGHT, no equivalent mutations: `swap-forget`, `g1-map-flip`, `tie-strict` (or → and), `b-first`, `error-as-tie`, `adj-no-half` (half-win → full win), `denom-short`, `c-as-b`, `no-mark-tie`, `case-insensitive`, `absent-skip`, `half-zero-to-one`.

## Hand-computed anchor pitfalls

- The oracle's parse was first written as "position of the mark in the text wins" (min(hits)) — the official semantics is mark priority; the cross-check failure exposed it.
- In the aggregation case, match 12's g2_judgment was originally "whatever" (no marks) → error-drop takes priority over the tie rule and the whole match vanished — the hand computation missed the error priority; this instead produced the dedicated check_error_beats_tie, the most subtle ledger point of this batch.

## Boundaries with adjacent problems

- `eval-ifeval-rules` (order 665): instruction-following rule verifier (deterministic rules, no judge model).
- This problem: the **evaluation-protocol ledger** of LLM-as-judge (position-bias countermeasure and aggregation), not the judge model call itself.
- Follow-up candidates: single-mode scoring (TIE_DELTA + score=-1 dropping), the [[a,b]] two-score regex backup, compute_agreement's kappa ledger.
