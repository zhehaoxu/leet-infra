# notes (rs-ttrl-pseudo-reward-ledger)

## Sources

- Paper: TTRL (arXiv 2504.16084, NeurIPS 2025). The claim is "vote at test time to mint a
  label": draw `n` samples for one question, vote for a pseudo-label, then use it as ground
  truth to produce 0/1 rewards for RL. The paper gives the pipeline and the curves; the
  **voting details, ratio denominator and metric definitions** are not written down.
- Code: `PRIME-RL/TTRL` main (the official implementation, a verl fork).
  - `verl/verl/trainer/ppo/ttrl_utils.py`: 20-32 `select_top_k_per_prompt`, 38-47
    `apply_original_gt`, 50-83 `apply_ttrl_gt`, 86-106 `_batch_majority_vote`, 109-122
    `_majority_vote`, 128-155 `compute_ttrl_metrics`, 158-190
    `_batch_compute_ttrl_metrics`, 192-214 `_prompt_compute_ttrl_metrics`.
  - `verl/verl/utils/reward_score/ttrl_math/__init__.py`: 30-33 `extract_answer`, 36-47
    `grade`, 49-65 `simplify_expression_string` (sympy), 67-103 `compute_score`.
  - `verl/verl/utils/reward_score/ttrl_math/math_utils.py`: 891-915 `last_boxed_only_string`,
    918-925 `remove_boxed`, 928-932 `extract_boxed_answer`.
  - `verl/verl/utils/reward_score/ttrl_math/cluster.py`: 1-44 `TTRLClusterCounter` (a cluster
    counter taking an equivalence function) — note that **`_majority_vote` does not use it**;
    it uses a bare `Counter`, i.e. only verbatim-equal canonical answers share a vote.
    Equivalence-class aggregation is a path the repository leaves unwired.
- This problem externalizes the three undecidable pieces (LLM generation, sympy simplification,
  mathd/sympy grading) as inputs and keeps only the bookkeeping.

## Traps in the official implementation

1. **`majority_ratio` divides by `n`** (`majority_count / len(model_outputs)`), not by the
   number of successful extractions. A question where half the samples fail to parse gives a
   lone winner only 0.25.
2. **The nothing-parsed fallback is the string `"None"`** with ratio `0.0`. It is not `None`,
   and it flows straight into the later `label_accuracy` comparison — so a ground truth that
   happens to be `"None"` scores 1.0.
3. **Ties go to the earliest occurrence**: `Counter.most_common(1)` is
   `sorted(..., key=count, reverse=True)`, and Python's stable sort over an insertion-ordered
   `Counter` means the first-seen answer wins.
4. **Only the last `\boxed` counts**: `last_boxed_only_string` uses `rfind`, so a mangled final
   box ruins the whole sample and an earlier well-formed box does not rescue it.
5. **The `\fbox` branch is dead code**: `extract_answer` already rejects text without a `\boxed`
   substring; and even if `last_boxed_only_string` were called directly, `remove_boxed` only
   accepts the `\boxed{` prefix, so `\fbox{7}` always yields `None`. The problem therefore only
   says "`\fbox` does not count".
6. **Brace balancing starts at the `\boxed` position**, so a `}` appearing before any `{` drives
   the depth negative and it never returns to 0; a space after `\boxed` with braces further down
   balances fine but fails the prefix check. Both are `None`, for different reasons.
7. **An empty box is a legal vote**: `\boxed{}` extracts `""`, `simplify` fails and returns `""`
   too, so the empty string can win and can be graded correct.
8. **`select_top_k_per_prompt` never checks `k <= n`**: `range(start, start + k)` bleeds across
   blocks and the last block runs off the end with `IndexError`. The pipeline afterwards treats
   `k` as the new `n`, so every denominator moves with it.
9. **`reward_accuracy` counts sample-wise equality**, not equal means: it is the number of `i`
   with `majority_reward[i] == gt_reward[i]` divided by `n`.
10. **`pass@{n}` uses `sum(gt_reward) >= 1`**, not `any`; for 0/1 rewards the two agree, but the
    form is a sum.
11. **The cross-question average takes its key order from the first question**
    (`for k in ttrl_metrics[0]`), and `majority_ratio` is appended afterwards, so it always sits
    last.
12. **The "all `n` labels in a question are identical" assert in `_batch_compute_ttrl_metrics` is
    constructively redundant**: `apply_ttrl_gt` writes the same label into all `n` samples. This
    problem does not replicate that dead assertion.

## Normalizations in this problem

- **A table replaces sympy**: the official `simplify_expression_string` tries
  `parse_expr(transformations="all", evaluate=False)`, falls back to `latex2sympy`, then returns
  the raw string. Here the result is a `canon` table whose misses stay verbatim (equivalent to
  both fallbacks failing). Every raw value is looked up once, so chained tables do not chain.
- **String equality replaces mathd/sympy grading**: the official `grade` is
  `grade_answer_mathd or grade_answer_sympy` (the latter also does `_normalize`, `split_tuple`
  and `are_equal_under_sympy`). Without sympy in the judge environment this degrades to
  "canonical equality", the single **semantic** downgrade in the problem; every other rule is
  unchanged.
- **An unextractable ground truth falls back to False**: the official `grade` would raise
  `TypeError` inside mathd when handed `None`; here it is normalized to "no match".
- **An empty batch raises `ValueError`**: upstream hits `IndexError` at `ttrl_metrics[0]` inside
  `_batch_compute_ttrl_metrics`; this problem checks up front and raises `ValueError`.
- **The unreachable divisibility re-check was deleted**: after downsampling the length is always
  `questions * keep` and `n` equals `keep`, so the second "length divisible by n" check always
  holds; it is gone (an equivalent mutation of it survives trivially).

## Mutations

45 mutations: **39 substantive ones all CAUGHT**, 6 judged equivalent, each cross-checked by
replaying 3000 random exam sheets (with `keep`, empty boxes, broken ground truths and chained
canon tables) plus 30000 random strings against the ledger with 0 differences:

- `m-depth-le` (`depth == 0` → `depth <= 0`): the depth only goes negative when a `}` precedes a
  `{`, and the sliced candidate then starts with `\boxed}`, so the prefix check returns `None`
  either way.
- `m-end-check` (`end < 0` → `end == 0`): on failure `end` is `-1` and `text[idx:0]` is the empty
  string, which also fails the prefix check; on success `end >= 1` is never 0.
- `m-truth-item-none-only` (dropping half of the `truth is None` condition): with a `None` truth
  the comparison becomes `canon(answer) == None`, and a string never equals `None`, so the result
  stays 0.
- `m-mvr-div-len` (`sum(rewards) / n` → `/ len(rewards)`): `rewards` has length `n` by
  construction, so both denominators are the same number.
- `m-pass-ge-half` (`>= 1` → `>= 0.5`): ground-truth rewards are only 0/1, so both thresholds
  agree.
- `m-metrics-first-last` (averaging key order from `prompts[-1]` instead of `prompts[0]`): every
  question has exactly the same metric keys in the same order, so nothing is observable.

Two that survived the first round and were brought back:

- `m-div-guard-off`: short-circuiting the pre-downsampling divisibility check was originally
  covered by a **second** re-check, which was dead code I had written. Adding a "not divisible
  plus a `keep`" case (upstream must assert first, otherwise the tail is silently dropped) and
  then deleting the dead code finally catches it.
- `m-scan-after-prefix` (`idx` → `idx + 6`): `\boxed` is exactly 6 characters, so `+6` lands on
  the opening brace and the mutation is vacuous; `+7` (skipping the brace) is the meaningful one.

## Overlap check

- Versus `rs-self-consistency-vote` (order 654): that problem receives **answer spans** and tests
  eight lexical normalization rules plus weighted vote counting; this one receives **raw
  completions** and tests `\boxed` extraction (rfind + brace balancing + prefix check),
  pseudo-labels, the two reward paths, the metric definitions and downsampling — normalization is
  an input here.
- Versus `rs-prm-best-of-n` / `rs-budget-forcing` / `rs-overthinking-metrics`: step-level
  aggregation, a budget state machine and length statistics respectively; none touches
  pseudo-labels or reward accounting.
- No TTRL / pseudo-reward / `\boxed`-extraction problem exists elsewhere in the repo; the voting
  problems under `agentic-systems` (ToT, CoA) schedule trees and chains, with no "mint a label
  from your own samples, then score against it" axis.
