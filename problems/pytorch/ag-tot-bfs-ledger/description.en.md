# Tree-of-Thoughts BFS Search Ledger

## Problem

Tree of Thoughts (Yao et al., 2305.10601) replaces single-chain CoT with tree search: at each step, multiple candidate thoughts are expanded from the current states, scored by an evaluator, and only the best b survive. The official implementation (princeton-nlp/tree-of-thought-llm) puts this loop in `src/tot/methods/bfs.py::solve`, paired with the Game-of-24 value head (`src/tot/tasks/game24.py`).

Generation and evaluation are non-deterministic LLM calls, so this problem takes them **entirely as explicit inputs** and tests only the search loop itself:

- **Expansion**: in `propose` mode each model output splits on `\n` into proposals, children = `y + p + "\n"` (official appends a **trailing newline**); in `sample` mode children = `y + s` (**no separator**). Candidates flatten parent-major (itertools.chain semantics); surplus parent slots beyond the actual parent count are ignored.
- **Scoring** (game24 value head): each evaluation text contributes its **last line**, matched exactly against the weight table `{impossible: 0.001, likely: 1, sure: 20}` (officially marked "ad hoc"), summed **without dividing by the sample count** (two "sure" lines score 40, not 20); a candidate of exactly 4 lines with no `answer` (case-insensitive) scores 0; **a candidate repeated within the same step scores 0** (the official `get_values` local-cache quirk — the second occurrence is not a cache hit, it is hard-zeroed, its evaluation texts ignored).
- **Selection**: greedy stable descending sort keeping the top b, **ties keep original index order** (lower id first).
- **Carry-over**: this step's `select_new_ys` becomes the next step's parents.

Emit the per-step ledger `step / x / ys / new_ys / values / select_new_ys`, with the last selection as `final`.

## Solution

Implement `solve(x, plan, n_select_sample=5)`. Domain constraints: `n_select_sample` is a positive int; plan is non-empty and each step carries `method/gen/evals`; unknown method, missing slot, or `evals[i]` length mismatching the parent's child count → `ValueError`.

## Example

```python
solve("1 2 3 4", [
    {"method": "propose", "gen": ["a\nb"],
     "evals": [[["sure", "sure"], ["likely", "impossible"]]]}
], n_select_sample=5)
# new_ys = ["a\n", "b\n"] (propose appends trailing newline)
# values = [40.0, 1.001] (2x sure, no normalization; 0.001+1)
```

## Constraints

- `1 <= steps <= 3`, at most 5 parent slots per step, `b in {2,3,5}`; candidates/eval texts are short strings.
- 200 seeded random cross-checks plus exact anchored assertions.
