# Tournament-GRPO: Knockout Tournament Reward Ledger (789)

**Source**: datawhalechina/agentic-rl-lab `experiments/tournament-grpo` (adapted from PuffYang/Tournament_GRPO).

## Problem summary

LLM-as-judge scoring a whole group at once is unstable — same-question candidates play a two-player knockout: the judge sees only 2 answers per match and picks one winner; win counts become the reward. Transcripts first pass a binary 0/1 format gate (stack-based tag parsing + structure validation); total = min-max normalized tournament reward + format reward. The LLM judge is parameterized as a (seed, pick) deterministic strategy with random.Random(seed) driving the draws — fully deterministic.

## What to solve

`solve(mode=..., ...)` with six entries (mode kwarg dispatch, stdlib only):

1. `mode="parse"` (text) → [top-level tag blocks, sorted error set]
2. `mode="answer"` (text) → strict trailing `<answer>` extraction
3. `mode="format"` (text) → {format_reward, failure_reasons, rollout_answer}
4. `mode="normalize"` (scores) → group min-max (all-equal → 0.5 each)
5. `mode="tournament"` (query, answers, seed, pick) → {raw_scores, rewards, matches}
6. `mode="score"` (query, solutions, seed, pick) → {results, log}

## Key official semantics (pinned against the source, see notes.en.md)

- Stack parsing: a mismatched close tag falls back to the nearest same-name opener, silently dropping the in-between unclosed tags
- Errors deduplicated as a set, emitted in the fixed TAG_ERRORS order; only call_tool may carry attributes and must carry name
- `<answer>` four gates: unique, non-empty, tag-free, nothing after the closing tag
- Structure: non-empty think first → each segment has ≥1 google_search cycle plus any browse_webpage cycles → answer
- Tournament: M=4 repeats sharing one rng, G=2 per match, round-robin grouping `candidates[i::num_groups]` (odd counts yield a 3-player group), winner +1 and advances
- No-answer candidates enter with "" and get their tournament reward zeroed **after** normalization (still shaping others' min-max range)
- Total ∈ [0, 2]: tournament reward + format reward

## Scoring

- A set of `check_*` functions; EXPECTED values dumped from the official reward.py sandbox (/tmp/tg_check.py)
