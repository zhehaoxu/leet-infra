# ADaPT As-Needed Decomposition Ledger

ADaPT (As-Needed Decomposition and Planning, arXiv 2311.05772) argues for **trying first and
decomposing only on failure**: the executor attempts the whole task directly, and only when that
fails and the depth budget is not exhausted does the planner split it into `AND` / `OR` subtasks
that are then solved recursively.

This problem strips `archiki/ADaPT`'s `run_alfworld.py` down to a pure function: LLM calls become
table lookups, the environment becomes a script, and only the **control flow and the ledger**
survive — who attempted what at which depth, how a failure got decomposed, how `AND` / `OR`
short-circuit, and how actions and information travel between nodes.

Implement `solve(task, exec_table, plan_table, max_depth=3, max_runs=20, max_patience=5, info_prop_mode="last-step-last-act")`.

## Input

- `task`: root task text (non-empty string).
- `exec_table`: executor scripts, `{task: {"steps": [{"act", "obs", "reward", "done"}, ...]}}`.
  `obs` defaults to `"Nothing happens."`, `reward` to 0 and `done` to `False`.
- `plan_table`: planner outputs, `{task: plan_text}`.
- `max_depth`: depth budget; the root has depth 1.
- `max_runs`: action cap for a single attempt.
- `max_patience`: threshold of consecutive no-progress observations.
- `info_prop_mode`: `"last-step-last-act"` or `"all-step-last-act"`.

## Output

```
{
  "nodes": [ one record per visited node, in pre-order ],
  "runs": action turns + planner calls,
  "checkpoint": real actions of the successful path,
  "completion": execution text accumulated at the root,
  "info_prop": information string computed at the root,
  "success": bool, "reward": int, "terminate": bool,
  "depth": depth returned by the root,
}
```

Node fields: `id` (pre-order), `task` (a task string, or the **already parsed sub-plan** for a
parenthesised group), `depth`, `replayed` (checkpoint length replayed on entry), `direct` (the
direct attempt: `success` / `reward` / `terminate` / `turns` / `history`; `None` for groups),
`plan` (parsed `{"steps": [...], "logic": ...}`, `None` for leaves), `children` (child ids),
`stop` (`"success"` / `"depth-cap"` / `"and-fail"` / `"or-success"` / `"children-done"`),
`success`, `reward`.

## Executor semantics

One attempt at a task (`direct`):

- Walk the script from step 1 and execute **at most `max_runs - 1` steps** (upstream writes
  `range(1, max_runs)`).
- An action starting with `think:` forces the observation to `"OK."`; the body is checked
  case-insensitively for `task completed!` (success) and `task failed!` (failure). Think actions
  **never enter `history`** but still count toward `turns` and `completion`.
- Other actions enter `history`.
- An observation in `{"Nothing happens.", "OK."}` increments the patience counter, anything else
  resets it; when the counter equals `max_patience` the run aborts (`terminate = True`) and **the
  step that triggered it is not written into `completion`**.
- A non-zero `reward` sets success and clears `terminate`.
- A truthy `done` ends the run immediately.
- When the loop drains without `done`: success is forced `False`, `terminate` becomes `True` and
  **`reward` drops to 0**.
- `completion` concatenates `" {act}\n{obs}\n>"` per step.

## Recursion semantics

For a string task:

1. Run the executor once (counted in `turns`).
2. If it succeeded, or `depth >= max_depth`, return. The returned text appends
   `completion + "\n"` — note that this happens **even on failure**, so a failed attempt at the
   depth cap still carries its text upward.
3. Otherwise look up `plan_table` (a planner call adds 1 to `runs`) and parse the plan; a single
   top-level subtask is unwrapped (a bare string becomes `{"steps": [x], "logic": "AND"}`), then
   `depth` increases by 1.

For a dict task (a parenthesised group): no executor run, no planner call, **no depth increase** —
expand its `steps` / `logic` directly.

The subtask loop:

- `logic == "or"` and the child succeeded → return immediately (`or-success`).
- Any successful child → append its `completion + "\n"` to the running text.
- `logic == "and"` and the child failed → return immediately (`and-fail`).
- All children done → return the last child's result (`children-done`).

Only the real actions of successful nodes are appended to `checkpoint` (pre-order).

## Plan parsing

Scan the plan line by line:

- Lines starting with `Step` are subtasks; every `Step <n>: ` prefix inside the line is stripped;
- A line containing `execution order` (case-insensitive) supplies the order, taking the **last**
  segment after splitting on `": "`;
- The expression lexer only recognises `Step <n>` / `AND` / `OR` / `(` / `)`, everything else is
  ignored; when several logic words appear at the same level the **last one wins**; parentheses
  produce nested sub-plans;
- Any exception during parsing (no order line, unbalanced parentheses, out-of-range `Step` index)
  falls back wholesale to `{"steps": [subtask text...], "logic": "AND"}`, losing the nesting.

## Information propagation

`info_prop` is carved out of `completion`: `rstrip("\n>")`, require the literal string
`Task completed!` (case-sensitive, unlike the executor's check), split into lines, drop the last
two, skip trailing think pairs, then take the final two lines (the `act` line has any
`Task completed!` occurrence removed).

`"last-step-last-act"` refreshes at every success (leaf, child, `OR` hit); `"all-step-last-act"`
only writes `past + "\n" + extracted` with leading newlines stripped when a **leaf** succeeds.

## Example

```
task   = "t"
exec   = {"t": {"steps": [{"act": "look", "obs": "Nothing happens."}]}}
plan   = {"t": "Step 1: wash the mug\nStep 2: put the mug\n"
               "Execution order: (Step 1 AND Step 2)"}
result = success=True, runs=6, checkpoint=["go to sink 1", "go to cabinet 1",
                                          "put mug 1 in/on cabinet 1"]
```

## Constraints

- Pure function: no network, no file I/O, no mutation of the inputs (deep-copy internally).
- A missing executor script, a missing plan when decomposition is needed, a plan that yields zero
  subtasks, `max_depth < 1`, `max_patience < 1`, an empty `task`, or an unknown
  `info_prop_mode` must raise `ValueError`.
