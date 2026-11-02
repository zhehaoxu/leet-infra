# notes (ag-adapt-decomp-ledger)

## Sources

- Repo: `archiki/ADaPT` main (the paper's official implementation). Core file `run_alfworld.py`
  (603 lines): `alfworld_run` (412-460, the executor loop), `plan_and_run` (476-561, the recursive
  decomposition itself), `plan_to_args` (462-473), `parse_expression` (381-400), `fetch_args`
  (402-410), `fetch_salient_info` (276-303). `TextCraft/env.py` and `run_textcraft.py` wrap the
  same algorithm around a different environment with identical semantics.
- Paper: ADaPT (arXiv 2311.05772, NAACL 2024 Findings) §3 describes "executor tries, planner
  decomposes as needed, recurse", but says nothing about when the depth cap is checked, how the
  ledger looks after an AND/OR short-circuit, or what happens when parsing fails. Those details
  live only in the code and are the main source of this problem.
- Only the control flow is ported: LLM calls become table lookups, the environment becomes a
  script, and `env.reset()` + checkpoint replay becomes the ledger's `replayed` field.

## Official quirks (verified one by one)

1. **As-needed means try first, then check depth**: `if succ or depth >= max_depth: return` sits
   *after* the executor run, so `max_depth=1` never decomposes at all — the root still tries once.
2. **`max_depth` counts decomposition levels, not execution levels**: the root has depth 1 and
   `depth += 1` happens only after a successful plan, so `max_depth=2` lets only the root decompose.
3. **`range(1, max_runs)` runs one step short**: the loop starts at 1, so `max_runs=20` gives at
   most 19 steps and `max_runs=1` executes nothing.
4. **Patience uses `==`**: `if pat_ctr == max_patience`, and only `"Nothing happens."` / `"OK."`
   count as no-progress; anything else resets the counter. The triggering step breaks before
   `prompt += ...`, so it never reaches the completion, though its action is already in
   `action_history`.
5. **A drained loop wipes the books**: `if not done: success = False; terminate = True` and the
   function returns the literal `0`, discarding any accumulated reward.
6. **Reward can be undone**: `if reward: success = True; terminate = False` runs first, but a later
   patience abort or exhausted script forces success back to `False`.
7. **Think actions have three effects**: the observation is forced to `"OK."` (so it counts toward
   patience), it never enters `action_history`, and it does enter `prompt` and `num_runs`. The
   completed/failed checks are substring tests over `action.lower()`.
8. **A parenthesised group is an already-parsed plan**: the non-string branch of `plan_and_run`
   does `plan_steps = task` directly — no executor, no planner, no `depth += 1` — so a group node
   shares its depth with its siblings.
9. **Parse failures fall back wholesale**: the `try/except` in `plan_to_args` swallows anything from
   `parse_expression` + `fetch_args` and returns `{'steps': args, 'logic': 'AND'}`. The fallback
   hands back the **raw strings**, so parenthesised nesting disappears; an out-of-range `Step`
   index, an extra closing paren or a missing order line all take this path.
10. **The last logic word at a level wins**: `current['logic'] = token` is an assignment, so
    `Step 1 AND Step 2 OR Step 3` makes the whole level `OR`; the lexer only knows uppercase
    `AND`/`OR`, and a lowercase `or` is silently dropped as noise.
11. **Single-step unwrapping**: when `len(plan_steps['steps']) == 1` the sole element is lifted; a
    string becomes `{'steps': [x]}`, a dict is kept as-is. The upstream `if 'logic' not in
    plan_steps.keys():` guard is inverted (inside that branch `plan_steps['logic']` always raises
    KeyError), so the effective behavior is "default to AND when missing".
12. **Completion only accumulates successes — except at the depth cap**: `running_completion +=
    completion` sits outside `if succ` on the depth-cap path, so a failed leaf still carries its own
    text back up.
13. **The checkpoint subset test is dead code**: `if not set(act_history).issubset(action_checkpoint)`
    compares two names bound to the same list (already extended by the successful leaf), so it is
    always true and the guarded extend never doubles anything.
14. **`fetch_salient_info` is string arithmetic**: it requires the literal `Task completed!`
    (case-sensitive), does `rstrip('\n>')`, drops the last two lines via `[:-2]` and then skips
    trailing think pairs. When the text ends with a think pair this lands exactly on the last real
    action; when it ends with a real action the window **slips** (yielding values like `OK.` and
    `>`), and every `act` line from the second segment onward still carries the leading `> `.
15. **Info extraction is case-sensitive while success detection is not**: the executor tests the
    lowercase `task completed!`, the extractor the literal `Task completed!` — a lowercase
    completion yields `success=True` with `info_prop=""`.
16. **`all-step-last-act` is effectively always empty**: the child-success branch only handles
    `last-step-last-act` (there is no else), and the all-step branch of `_update_info` only fires on
    leaf success, where `past_info` is always `""` (see equivalent mutation 1).
17. **`env.reset()` plus replay**: every node resets the environment and replays
    `past_action_checkpoint`, yet those replayed actions never count toward `num_runs`; the ledger
    records their length as `replayed`.

## Equivalent mutations (with proofs)

1. `m-info-always-last` (making `_update_info`'s all-step branch behave like last-step) is
   **unobservable**: in all-step mode the subtask loop's `info` update is guarded by
   `if self.mode == "last-step-last-act"`, so every node receives `past_info == ""`. Then
   `(past + "\n" + x).lstrip("\n")` is character-for-character equal to `x`, and the root never
   updates its own `info` either — the only exception is a root that is itself a leaf, where
   `past_info` is `""` as well. So the branch is equivalent on every reachable path.
2. `m-unmatched-paren-ok` (keeping the current node instead of letting `stack.pop()` raise on an
   extra closing paren) is **unobservable**: keeping it makes
   `current.setdefault('steps', []).append(closed)` insert the node into its own `steps`, so
   `_fetch_args` recurses into a self-reference and raises `RecursionError`, while upstream raises
   `IndexError` from popping an empty stack. Both are swallowed by the bare `except` in
   `plan_to_args`, and both end in the same "flat + AND" fallback.

## Mutation stats

- All 37 substantive mutations CAUGHT, covering: depth-cap timing (`>=`→`>`), the one-short
  `range(1, max_runs)`, patience `==`→`>` and counter reset, think into/out of history, think
  case-folding, reward clearing `terminate`, wiping reward/success on a drained loop, the
  completion separator, dropping parens from the lexer, logic-word precedence, the paren append
  target, the order-line match (substring vs line start), the fallback logic, the unwrap threshold,
  whether groups bump depth, `OR`/`AND` short-circuits, the info mode and its case sensitivity, the
  `[:-2]` slice, stripping `Task completed!`, checkpoint appends, `replayed`/`children` ids, `runs`
  accounting, the `stop` label, the missing-script / empty-plan / mode validations, and all three
  defaults.
- 2 equivalent mutations (above), each with a structural proof.

## Engineering lessons

- **Trace before pinning assertions**: the `[:-2]`-plus-skip-think arithmetic in
  `fetch_salient_info` slips into values like `OK.` and `>` when the text ends with a real action.
  The first draft of the test asserted "the last real action" and failed outright; printing the
  completion line by line revealed the actual rule, and both shapes (ending with a think pair vs
  ending with a real action) are now pinned separately.
- **The root completion always carries a trailing newline**: the root does `running += completion
  + "\n"`, so even a zero-step run returns `"\n"` rather than `""`. "Empty is not the empty string"
  details like this only show up in a trace.
- **Fixtures must separate "incidental" from "specified"**: giving 'a'/'b'/'c' the same script
  (all `go to a 1`) makes the checkpoint hold three identical entries and hides who wrote them;
  distinct action names make the assertion informative.
- **Error cases must dodge earlier failure points**: pairing `info_prop_mode="nope"` with an input
  that already lacks a plan keeps raising `ValueError` even after the mode check is deleted (the
  missing plan fires first), so that mutation survived. Switching to an input that would otherwise
  succeed caught it immediately.
- **Oracle and reference solution should share semantics but not structure**: the oracle uses a
  `while` cursor, recursive descent and a flat ledger, while the solution uses a `_Runner` class and
  a stack-based parser; 40 randomized scenarios cross-check both implementations.
