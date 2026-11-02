# notes (ag-lats-uct-ledger)

## Sourcing

- Official repo **lapisrocks/LanguageAgentTreeSearch** (originally andyz245/LanguageAgentTreeSearch; lats-group/lats is 404). The MCTS core is `programming/mcts.py` (isomorphic copies under hotpot/webshop). **Paper number correction: LATS is arXiv 2310.04406** (Zhou et al., ICML 2024) — the todos candidate wrongly said 2407.01476; fixed in sync.
- File tree: `programming/{mcts,dfs,reflexion,immediate_refinement,main}.py` + executors/generators. The full `run_mcts` main loop (~230 lines) was captured.

## Key official quirks (verified line-by-line)

1. **UCT visits==0 returns value** (`#return float('inf')` commented out) — unvisited nodes have uct=value=0. Corollary: **at most one sibling per level is ever visited** — each iteration only the leaked/break child gets a direct update, and backprop follows a single path; unvisited siblings have uct=0 and are dominated by any visited sibling with reward≥0 (or resolved by tie-first order). ⇒ **UCT mean-vs-sum is unobservable within the official flow** (eq mutation verified SURVIVED; noted) — reward is always ≥0 (count/n_tests ≥0, real∈{0,1}), so mean and sum share sign and the unvisited sibling is always 0.
2. **The leaked loop variable decides the update object**: after `for child in node.children: ... break`, `child.update(reward)` — the break child on break, otherwise the last child; **the other children of the batch get zero updates** (the official code never updates them).
3. **Selection always descends to a leaf** ⇒ the inner `for child in node.children` actually iterates only the n newly added children (historical children are never re-simulated) — the official "rerun all children" form is harmless in this flow. The first test draft wrongly assumed historical children were re-simulated; instrumentation disproved it.
4. **Last-iteration quirk**: `if is_passing_internal or cur_iter == max_iters - 1:` is true for the **first** child → only the first child is simulated before evaluate+break. All iterations=1 cases take this path — the first anchor set assumed "simulate all" at iterations=1 and broke.
5. **Reward counting**: `"Tested passed:"` (Tested!) vs `"Tests failed:"` (Tests) spelling mismatch; `split("Tested passed:")[1]` takes the tail; non-empty means `line.strip() != ''` (whitespace-only lines don't count); without the marker reward=0 (denominator len(tests_i) is the internal test count).
6. **gather_context_from_tree**: collect along parents, `[::-1]` reversed (earliest first); called on the **selection endpoint** (the new child's own feedback excluded). The root's first-attempt feedback/reflection are collected too — hence the root_feedback/root_reflection inputs.
7. **Terminal best_child_value**: `max(children, key=lambda c: c.value)` — **cumulative value, not the mean**; a childless root → None.solution AttributeError → ValueError.
8. **A failed eval still breaks**: inside `if is_passing_internal or last_iter:` the break is **unconditional** after evaluate — eval False gives reward_real=0 but the loop still terminates.
9. **Backprop reaches root**: `temp = child; while temp.parent: temp = temp.parent; temp.update(reward)` — child already updated outside the loop; the chain from parent to root all +1 (root has no parent but is included).

## Authoring engineering

- Simulation results are `(passing, feedback, reflection, eval)` quadruples queued in the official **execution order** (one consumed per child executed); eval is consumed only on the break round — a generous queue suffices.
- **The ledger's updated_child semantics was the biggest trap**: the first solution/oracle both wrote the update object as the pre-initialized `children[-1]`, losing break semantics — when oracle and solution share the same bug, _compare cannot notice it; only an independently hand-computed anchor (break_on_old_child) caught it. Lesson: oracle-solve shared errors must be backed by independent manual anchors.
- **The oracle's gather forgot the [::-1] reversal** — random cross-check case 13 caught it (acc_feedback order flipped). The oracle is not ground truth; line-by-line comparison against the official source is the only standard.
- Node terminal output: solve pops parent_ref in place (nested children share the objects, so they clean up together); the oracle must strip pref **recursively** — stripping only the top level leaked the nested key, caught by random as a {'pref'} key diff.
- 18 substantive mutations all CAUGHT + 2 equivalents (sum-not-mean, passing-keep-refl — the latter is officially redundant anyway: fresh children have reflection "" by default). **An eq mutation was once misjudged CAUGHT**: the flip anchor's own expectation was wrong (I misread two levels of selection debug output as one, and treated mid-run state as terminal) — after fixing the anchor the eq mutation returned to SURVIVED. Lesson: prove a check passes on the correct implementation before using it to judge mutations.
- Red lines: prompt templates, gen.func_impl retry loops, exe.evaluate sandboxing, prune_context_blocks (the split('Previous Trial')/join('trial') delimiter-mismatch bug is delightful but the pruning is dead code outside the mcts.py main loop — excluded), pass@k statistics.

## Future candidates

- The hotpot LATS variant (ReAct-style action space + wikienv) and the webshop variant could make comparison problems; dfs.py's DFS ledger (vs MCTS selection differences) can be folded into the ToolLLM DFS candidate.
