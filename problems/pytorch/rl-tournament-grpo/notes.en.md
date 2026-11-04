# Implementation notes (789 rl-tournament-grpo)

## The official ledger, line by line (reward.py)

- `parse_transcript`: scans TAG_PATTERN tag by tag; `cursor` tracks the end of the previous tag — stray text outside tags while the stack is empty → stray. Open tags: parse_attrs validates, then push; close tags: same-name stack top → pop; otherwise broken_xml + fall back to the nearest same-name opener (`max(i for ...)`, default=None → the close tag is ignored), `stack = stack[:index]` dropping the in-between unclosed tags. Blocks are produced only when the stack is empty (`if not stack: blocks.append(...)` — the top-level semantics). At the end: stack empty with trailing text → stray; stack non-empty → broken_xml.
- `parse_attrs`: stray text between attributes or duplicate keys → (attrs, False); call_tool's allowed set is {name}, others empty; `valid = no trailing text and no out-of-set attrs`, and call_tool additionally needs name.
- `extract_answer`: exactly one match, non-empty after strip, tag-free content, no text after the closing tag.
- `check_structure`: blocks[0] is a non-empty think; len ≤ 2 → no tool cycle; main loop — each segment must start with is_tool_cycle(google_search), search cycles exhausted then browse cycles exhausted, back to while.
- `analyze_format`: failures = TAG_ERRORS ∩ errors (fixed order) + tool:unknown_tool_name (block scan) + answer:invalid_or_missing; **structure runs only when everything else is clean** (official comment: the answer is then unique and last, so must_end_with_answer / invalid_tool_cycle_order can never trigger — the author omits them).
- `normalize_scores`: min-max; high==low → all 0.5.
- `run_tournament`: `rng = random.Random(judge.seed)` created **outside** the 4 repeats (shared); each round shuffles → `num_groups = len//2` → round-robin `candidates[i::num_groups]` (odd counts yield a 3-player group) → judge per group → winner +1 and advances; matches record repeat/round/candidates/winner/output.
- `score_group`: formats → answers (truncated at 120000) → tournament → `reward = reward if answer else 0.0` (**zeroed after normalization**) → score = tournament + format.

## Sandbox engineering

- /tmp/tg_check.py (python3, stdlib only): stubs data.ResearchExample / judge.JudgeClient; the judge replaced by a (seed, pick) deterministic strategy (longest = longest answer in the group, first = index 0).
- **Lesson**: pick=first's 3-player-group winner is the post-shuffle group head and varies with the draw — never hand-assert [4,0,0]; dump the official ledger instead (tour3_first's raw is also [1,3,0]).

## Judging protocol

The judge injects a single `solve`; `mode` kwarg dispatches six entries. `check_*(solve)` assertion set, 6 cases (stdlib only, no torch dependency).
