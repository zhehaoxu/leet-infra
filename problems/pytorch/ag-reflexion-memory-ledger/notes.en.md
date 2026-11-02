# notes (ag-reflexion-memory-ledger)

## Sourcing

- Official repo **noahshinn/reflexion** (default branch is `main`, not master). The memory pipeline lives in three places: `alfworld_runs/env_history.py` (EnvironmentHistory), `alfworld_runs/generate_reflections.py` (_get_scenario/_generate_reflection_query/update_memory), and the trial loops in `alfworld_runs/alfworld_trial.py` / `webshop_runs/webshop_trial.py` (the construction-time window). The hotpotqa side (agents.py's reflections_str assembly + truncate_scratchpad) needs tiktoken — not included.
- **Todos correction**: the candidate note said "episodic window max 5" — wrong. No 5-entry window exists anywhere in the repo; both the construction-time cut and the reflection-query cut use **3** (`[-3:]` if `len > 3`). Pinned to the source.
- **The two env_history.py copies disagree**: webshop's `_get_base_query` uses a single-newline section header, alfworld's uses double (plus a `# NOT CURRENTLY SUPPORTED` comment). Alfworld (the paper's main benchmark) wins; the webshop delta is recorded in the description.

## Key official quirks (verified line by line)

1. **Window is 3, not 5** (two independent code sites: construction + reflection query); len==3 is exactly not truncated; Trial numbering restarts at 0 over the truncated list.
2. **add exhaustion**: `value == self._last_action` — not adjacent dedup but comparison against the **most recent distinct action** (repeats across an observation trigger); sticky True; **the repeat branch does not update _last_action** — unobservable due to stickiness (equivalent mutation SURVIVED: on dup value==last_action so the assignment is a no-op).
3. **reset clears history only**: `_is_exhausted` and `_last_action` survive — repeating an old action after reset still exhausts (hand-computed anchor).
4. **Rendering**: starts with `cur_query + '\n'`; entries joined by `'\n'`, no trailing newline; human_edit prefix `'[human edit]: '` (officially commented NOT CURRENTLY SUPPORTED yet supported by both add and __str__ — live code).
5. **Trial-log split**: run_trial entries are `'\n#####\n\n' + body + '\n\n#####\n'`; adjacent seams (`'#####\n' + '\n#####'`) exactly form the separator `'#####\n\n#####'`; **measured**: parts count == env count (the assert's precondition); part0 = `'\n#####\n\n' + body0 + '\n\n'` (leading residual), middle parts `'\n\n' + body + '\n\n'`, the last part keeps the trailing `'#####\n'` plus the summary lines. All residuals flow into the scenario — a single-entry log's scenario ends with `#####`.
6. **Scenario**: `split('Here is the task:')[-1].strip()` — last segment on repeats; swallows the STATUS: FAIL tail (the official body is a full history string with trajectory and STATUS after the task line).
7. **Reflection gate** `not is_success and not skip`; the reflection appends to the **full** memory; the query re-truncates `[-3:]` (two independent windows).
8. **Two coexisting Trial formats**: base query `f'\nTrial {i}:\n{m.strip()}'` (no hash, stripped) vs reflection query `f'Trial #{i}: {m}\n'` (hashed, unstripped).
9. **webshop latent bug**: webshop's main.py initializes env_configs without a `'skip'` key while update_memory reads `env['skip']` — alfworld has it. Missing keys → ValueError here.
10. Official `env['memory'] += [reflection]` mutates env_configs in place; solve builds fresh dicts and returns them (equivalent ledger).

## Engineering

- Inputs: `ops` models the official reset explicitly as `("reset", "")`; `few_shot` (officially reflexion_few_shot_examples.txt) and the reflection header constant are explicit; the header is copied verbatim from the source (including 'Give your plan after "Plan"').
- **Test-input validity trap**: `''.split(sep)` yields `['']`, not `[]` — an empty log with 0 envs crashes the official assert; all sample checks use a valid 1-success-env + single-entry-log combination.
- **Hand anchors caught two of our own errors**: ① the A,A,B,A case — expected False, but the second A already exhausts (A==last A), comment and expectation both wrong; ② check_example initially missed the `#####` residual tail in the single-entry log's scenario. Lesson: split-residual quirks must be hand-computed character by character.
- Independent oracle: class-based EnvironmentHistory (segment list + join) vs solution's incremental `+=`; %-formatting vs f-strings to reduce common-mode errors.
- Mutations: 30 substantive all CAUGHT + 3 equivalences recorded: eq-lastaction-uncond (dup assignment is a no-op), eq-cap-history-thresh-2 (at len 3 `[-3:]` is the whole list, so thresholds >2 and >3 are equivalent), eq-queue-eq-check (qi increments by exactly 1, so `>=` equals `==`).
- Red lines: the reflection LLM call, the few-shot file contents, the hotpotqa scratchpad truncation (tiktoken), and webshop environment interaction are not modeled.

## Follow-up candidates

- The hotpotqa side's `format_reflections` / `format_last_attempt` / `truncate_scratchpad` (observation lines truncated one by one by token length down to 1600) as a sequel (tokenizer as an explicit input).
- webshop's `prompt[-(6400-len(init_prompt)):]` tail-window cut as a prompt-budget problem.
