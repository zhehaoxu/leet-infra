# LATS MCTS Search Ledger (ag-lats-uct-ledger)

**Difficulty**: Hard | **Collection**: agentic-systems | **Order**: 724

Stateless-ify the `programming/mcts.py` main loop of the official Language Agent Tree Search (LATS, arXiv **2310.04406**) implementation [lapisrocks/LanguageAgentTreeSearch](https://github.com/lapisrocks/LanguageAgentTreeSearch): the Node UCT selection, best_child tie rules, the reward flow of expansion/simulation/backprop, and several famously buggy official quirks (visits==0 returning value instead of inf, the leaked loop variable deciding the only child that gets a reward, the inconsistent `"Tested passed:"`/`"Tests failed:"` spellings) — each accounted. LLM generation and executor feedback are fully materialized as inputs.

## What you implement

`solve(root_solution, iterations, n, n_tests, plans, root_feedback="", root_reflection="")`: drive the MCTS loop iteration by iteration per `plans`, returning a per-iteration ledger, the final tree state, and the terminal choice.

## Ledger points (all verified line-by-line against the official mcts.py)

1. **UCT's visits==0 quirk**: `if self.visits == 0: return self.value` — the official code has `return float('inf')` **commented out**! An unvisited child's UCT is its value (default 0), not the traditional +inf.
2. **UCT formula**: `value/visits + 1.0*sqrt(ln(parent.visits)/visits)` — the logarithm takes **parent.visits** (not the common parent.visits+1); each node uses **its own parent's** visits.
3. **best_child ties**: `max(children, key=uct)` — Python max breaks ties toward the **first-extended** child.
4. **Selection always descends to a leaf**: `while node.children: node = node.best_child()` — the endpoint is always a leaf; hence the inner simulation's children are exactly the n newly added ones (the official `for child in node.children` "rerun all" form never touches historical children in this flow).
5. **Simulation consumption order**: each child consumes one `(passing, feedback, reflection, eval)` tuple from the queue; non-passing writes the reflection (overwriting), passing clears it; `test_feedback` is written unconditionally.
6. **reward_internal counting**: when feedback contains `"Tested passed:"` (official spelling), take the **non-empty (stripped) line count** after `"Tested passed:"` within the segment before `"Tests failed:"`, divided by n_tests; without the marker, 0.
7. **Break and the update object (official bug)**: on passing or the last iteration (`cur_iter == iterations-1`), evaluate the current child and break — the **child at the break point** is the update object; without a break, the leaked loop variable makes the **last child** the update object. The other new children of the batch get no reward at all (the official code never updates them).
8. **Last-iteration quirk**: on the final round the **first** child is evaluated and breaks — later children are never simulated.
9. **Reward/Backprop**: `reward = reward_internal + reward_real` (eval passing → reward_real=1 and solved); the update object updates first (visits+1, value+=reward), then the chain up **to the root** updates one by one. A failed eval still breaks (reward_real=0).
10. **gather_context_from_tree**: collect non-empty feedback/reflection along the parents of the **selection endpoint**, **reversed** to earliest-first — the new child's own feedback is not included.
11. **Terminal choice**: solved → best = the solution of the child that triggered it; otherwise `root.best_child_value()` — max by **cumulative value** (not the mean!), ties toward the first; a childless root raises AttributeError officially, converted to ValueError.
12. **Domain constraint**: zero-output expansion (no child to simulate) → ValueError.

## Example

```python
plans = [{"solutions": ["a", "b"],
          "sims": [(False, "err", "ra", None),
                   (True, "Tested passed:\nok", "", False)]}]
led = solve("root", 2, 2, 1, plans)
# iter0: c1 fails and continues, c2 passes -> evaluate(False) -> break
# updated_child == 2 (leak/break point), reward_internal == 1.0
# backprop == [0] (c2 self-updates first, then root)
```

## Scoring

12 checkpoints: the leaked update object, UCT visits==0, first-on-tie, last-iteration first-child evaluation, solved early exit, gather reversal and endpoint, cumulative-value terminal choice, reward counting (blank lines/spelling), backprop accumulation, domain constraints, plus 120 randomized MCTS-run cross-checks.
