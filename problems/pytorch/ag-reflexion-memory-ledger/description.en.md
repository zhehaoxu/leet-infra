# Reflexion Episodic Memory Ledger (ag-reflexion-memory-ledger)

**Difficulty**: Medium ｜ **Collection**: agentic-systems ｜ **Order**: 725

A stateless rebuild of the memory pipeline from the official Reflexion implementation ([noahshinn/reflexion](https://github.com/noahshinn/reflexion), arXiv **2303.11366**): `alfworld_runs/env_history.py`'s `EnvironmentHistory` (base-query triplet assembly, trajectory rendering, repeated-action exhaustion) and `alfworld_runs/generate_reflections.py`'s `_generate_reflection_query` + `update_memory` (trial-log splitting, reflection gate, the memory sliding window and the two coexisting Trial formats), line by line. LLM reflection outputs are explicit inputs. **Note: the official memory window is 3 entries (`[-3:]` when `len > 3`), not the often-cited 5.**

## What you implement

`solve(base_query, start_info, memory, ops, few_shot, env_configs, trial_log, reflections)`: assemble the EnvironmentHistory per the run_trial side (add/reset each op), then ledger the trial log per the update_memory side, returning both ledgers.

## Ledger points (verified line by line against the official source)

1. **Construction-time window**: `webshop_run`/`alfworld_run` build `EnvironmentHistory` with `memory[-3:] if len(memory) > 3 else memory`; the base query numbers `Trial {i}` over the **truncated** list starting at 0. len==3 is exactly not truncated.
2. **_get_base_query (alfworld version)**: `base_query` + (if memory non-empty) `"\n\nYour memory for the task below:"` + per entry `f"\nTrial {i}:\n{m.strip()}"` + `f"\nHere is the task:\n{start_info}"`. The webshop copy uses a **single** newline for the section header — the two run directories' env_history.py disagree; this problem pins the alfworld version.
3. **add's exhaustion quirk**: labels restricted to `action/observation/human_edit` (official assert → ValueError); an action **equal** to `_last_action` sets `is_exhausted=True` — repeats after an intervening observation also trigger (not adjacency-based), the flag is sticky True; the repeat branch does not update `_last_action` (unobservable due to stickiness — proven by an equivalent mutation).
4. **reset clears history only**: `_is_exhausted` and `_last_action` survive — a repeat of an old action after reset still exhausts.
5. **Rendering**: `cur_query + "\n"` then entries joined by `"\n"`, **no trailing newline**; action → `"> " + v`, observation → verbatim, human_edit → `"[human edit]: " + v`.
6. **Trial-log split**: `trial_log.split('#####\n\n#####')` — run_trial writes exactly one `"\n#####\n\n…\n\n#####\n"` entry per env; adjacent seams form the separator; `env_logs[i]` is env i's body; count mismatch with `len(env_configs)` → ValueError (official assert). **Residual-segment quirk**: part0 carries the leading `"#####\n\n"`, the last part carries the trailing `"#####\n"` plus the summary lines — both flow into the scenario.
7. **Reflection gate**: only envs with `not is_success and not skip` reflect; the reflection is appended back to the **full** memory (no truncation).
8. **Second sliding window**: the reflection query re-truncates with `[-3:]` if `len > 3` (independent code from step 1); older entries stay in memory but leave the query.
9. **_generate_reflection_query**: fixed long header + `":\n\n" + few_shot + "\n\n" + scenario`; with non-empty memory appends `"\n\nPlans from past attempts:\n"` and per entry `f"Trial #{i}: {m}\n"` (**hashed, unstripped** — coexisting with the base query's `Trial {i}` + `strip()` format); finally `"\n\nNew plan:"`.
10. **Scenario extraction**: `env_logs[i].split("Here is the task:")[-1].strip()` — **last** segment on repeats, and it swallows the STATUS/##### tail.
11. **Consumption order**: the reflections queue is consumed in reflected-env order; shortage → ValueError.
12. **Domain constraints**: missing any of the four env keys (the official webshop config lacks `skip` — a latent KeyError) → ValueError.

## Example

```python
mem = ["a", "b", "c", "d"]                       # len 4 > 3 → construction window ["b","c","d"]
ops = [("action", "A"), ("observation", "o"), ("action", "A")]
cfg = [{"name": "e0", "memory": ["p"], "is_success": False, "skip": False}]
log = "\n#####\n\nEnvironment #0:\nHere is the task:\nT0\n\nSTATUS: FAIL\n\n#####\n"
led = solve("Interact...", "task", mem, ops, "<fs>", cfg, log, ["r"])
# led["cur_query"] holds only Trial 0..2 = b,c,d
# led["is_exhausted"] is True (second A == last A, across an observation)
# led["queries"][0] contains "Trial #0: p\n" (reflection memory has 1 entry, uncut)
# led["envs"][0]["memory"] == ["p", "r"] (appended to the full memory)
```

## Scoring

12 checkpoints: the sliding window and Trial numbering, both base-query formats, exhaustion (cross-step repeats / reset stickiness), rendering joins, split residuals, the reflection gate and consumption order, the second window, the two Trial formats, last-segment scenario extraction, three ValueError classes, and 120 randomized memory/trajectory/log cross-checks.
