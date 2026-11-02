# ToolLLM DFS Decision Tree Ledger (ag-toollm-dfs-ledger)

**Difficulty**: Hard | **Collection**: agentic-systems | **order**: 728

Stateless-ify ToolLLM's (arXiv **2307.16789**) depth-first search-based decision tree (**DFSDT**) into a replayable tree ledger: given an LLM response queue, a tool-execution table and pairwise comparison votes, recompute the whole tree's expansion order, pruning / give-up / terminal bookkeeping, backtrack countdown and answer selection. Source: [OpenBMB/ToolBench](https://github.com/OpenBMB/ToolBench) master (the original X-PLUG/ToolLLM repo moved here): `toolbench/inference/Algorithms/DFS.py` (`DFS_tree_search.DFS`), `Tree/Tree.py` (`make_finish` / `compute_weight`), `LLM_rank/rank_candidate.py` (`rank2_subfix` / `rank2symmetry` / `sum_based_rankn`).

## What to implement

`solve(responses, tool_table, pair_votes, single_chain_max_step=5, tree_beam_size=2, max_query_count=20, answer=1, with_filter=True, give_up_seed=42)`: return the expansion/pruning/terminal ledger plus the final-answer ledger.

- `responses`: the LLM response queue, **one entry consumed per LLM query** (pre-order). Each item is `{"content": str|None, "function_call": {"name","arguments"}|None, "error_code": int, "tokens": int}`.
- `tool_table`: `{(action_name, action_input): (observation, status)}`; status: 0 ok / 1 hallucinated API name / 3 final answer / 4 give up.
- `pair_votes`: `{(i, j): [vote_ij, vote_ji]}` (i<j; i/j are candidate positions in `next_tree_split_nodes`); a vote is the raw `rank2_subfix` LLM output string.

## Ledger points (line-by-line source check)

1. **Entry-time triple test**: `DFS(node)` first does `expand_num = now_expand_num++`, then tests `depth >= single_chain_max_step or pruned or is_terminal`. The order sets priority: **terminal beats depth** (a status-3 node at the depth cap is still recorded as terminal), and **depth cap prunes** (`pruned = True`; only `observation_code == 4` becomes give-up and returns 2, otherwise 1).
2. **Backtrack countdown**: terminal returns `final_answer_back_length = 2`, give-up returns `prune_back_length = 2`, plain pruning returns 1; a parent receiving `result > 1` does `return result - 1` — the countdown says how many levels to climb back.
3. **DFSDT (`with_filter=False`)**: generate one child chain (Thought → optional Action + Action Input) and **recurse immediately**; a return value of 1 means no climb-back, so the loop keeps generating the next candidate at this level; `len(terminal_nodes) >= answer` returns 10000 and exits early.
4. **normal DFS (`with_filter=True`)**: first collect `tree_beam_size` child chains, then (when >1 candidate) rank them pairwise with the LLM: each pair is asked **in both directions** (`rank2symmetry`, `score[1-bigger1] += 1`, `score[bigger2] += 1`); the higher total earns 1 point, a tie splits 0.5 each (`sum_based_rankn`); expand in **stable descending** `prior_score` order (ties keep generation order). **Ranking queries count toward `query_count`** (2 per pair).
5. **Votes are decided by the last character only**: `rank2_subfix` uses `output["content"].strip().lower()[-1] == "a"` to mean "candidate 1 is better" — the wording of the string is irrelevant, only the final character counts.
6. **Backtrack inside the expansion loop**: `result > 1` (a terminal or give-up down the subtree) triggers `make_finish(2)` on the current node and `return result - 1`; the **remaining candidates at this level are never expanded** (even when `answer` is not yet satisfied).
7. **`make_finish(inter_val)`**: marks `finished` and propagates upward, decrementing on every `Action Input`-typed node (**itself included; the root is an Action Input too**), stopping once `inter_val < 0` — `make_finish(2)` stops at the second Action Input, so the third one and everything above stay unmarked.
8. **diversity candidates**: when a node already has children, walk each child's `children[0]` chain down to the **first** `Action Input` (stopping at a terminal or a leaf) and feed `{name: Action description, arguments: Action Input description, function_output: observation, value: compute_weight()}` back into the prompt; master's `compute_weight()` **always returns 0.0** (a leftover from ToT's Elo). An empty candidate list means **nothing is fed**.
9. **Budget**: `query_count >= max_query_count` is checked after counting and *before* node creation — the over-budget response is **discarded entirely** (no node built) and 100000 is returned, decremented by 1 per level on the way up.
10. **Answer selection**: scan `terminal_nodes` in append order for the **first unpruned** terminal; `final_answer` is that node's `description` (for an Action Input node, the action_input string). With no terminal, pick one of `give_up_nodes` at `randint(0, len-1)` (exposed here as `give_up_seed`).
11. **Domain constraints (ValueError)**: response queue exhausted; `tool_table` missing `(name, args)`; `pair_votes` missing a key or an empty vote string; **a response with no function_call arriving at an Action Input node** (the official code raises `new_message["function_call"]` KeyError here); `function_call` missing `name`/`arguments`.
12. **status side effects**: 4 → node pruned (give-up on entry); 3 → `is_terminal` + `make_finish(2)`; 1 (hallucinated API name) **leaves the tree untouched** and only rewrites the response's name to `invalid_hallucination_function_name` (message layer, not in the ledger).

## Example

```python
responses = [{"content": "try A first", "function_call": {"name": "get", "arguments": "A"}},
             {"content": "try B next", "function_call": {"name": "final", "arguments": "X"}}]
tool_table = {("get", "A"): ("okA", 0), ("final", "X"): ("ansX", 3)}
led = solve(responses, tool_table, {(0, 1): ["b", "a"]},
            tree_beam_size=2, single_chain_max_step=3, answer=1)
# last chars 'b'/'a' -> candidate 1 wins both directions -> expand its subtree (node 6) first
# expand_order == [0, 6]; terminal_nodes == [6]; finish_type == "give_answer"
# query_count == 4 (2 generations + 2 comparisons); answer=1 -> return_value == 10000
```

## Scoring

12 check points: DFSDT single-chain countdown, depth cap with terminal priority, the three give-up sources (tool status 4 / LLM error_code 4 / plain pruning), normal-DFS pairwise ranking and stable ordering (including the three-candidate 0.5 tie split and last-character-only votes), expansion-loop backtrack with `make_finish` span, dual-terminal registration order, budget discard, diversity candidates (empty list not fed, value pinned at 0), default arguments, domain constraints, plus a 150-script random cross-check.
