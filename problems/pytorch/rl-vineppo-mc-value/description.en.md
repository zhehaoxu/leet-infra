# VinePPO: Step-Level Monte-Carlo Return Accounting

## Problem

PPO-style value networks are hard to train on multi-step LLM reasoning: the
value head must predict, for every intermediate step, the expectation "will
we end up correct from here", with sparse and noisy supervision. VinePPO
(arXiv:2410.01679) drops the value network entirely: at each step branch,
continue-sampling k full rollouts, use final-answer correctness as the
reward, take each step's MC return mean as the value, and compute advantages
as a pure tree-shaped differencing. This problem replicates the official
`TreeEpisodeUtils` and all `PathAggregator`/`PathFilter` registered classes
(torch/Ray/multiprocessing stripped, accounting fully parameterized) via six
`solve(mode=...)` entries:

1. **score**: tree scoring. Leaf detection is `"answer" in node` (not the
   absence of children); the predicted answer is extracted from
   `node["answer"]` via `pred_map` — the node's own `"pred"` field plays no
   part; `None` → score 0, else `int(pred == gold)`; `is_correct_answer`
   redundantly mirrors score; internal nodes take the mean of child scores
   (0 if childless).
2. **advantage**: `child.advantage = child.score − parent.score`, a pure
   tree differencing (no bottom-up bias propagation); leaves return without
   assignment; root gets none.
3. **paths**: root-to-leaf path extraction. The tree is deep-copied first;
   `importance_weight = branch_factor ** (max_depth − depth)` but nodes at
   max_depth get 1; with `repeat_early_stopped_paths`, **every leaf
   shallower than max_depth** is extended importance_weight times; children
   are popped per node during traversal; a final json round-trip.
4. **aggregate**: `none` (as-is) / `unique` (dedupe via a set of
   json.dumps(sort_keys=True), output order unreproducible) /
   `top_k_percent` (sort by each path's mean/sum of
   `advantage.get("advantage", 0)`, reverse=True stable, `int(len*p)`,
   `keep_at_least_one` floors at 1) / `max_k_paths`
   (`random.Random(42).sample`).
5. **filter**: `successful` (last node's `is_correct_answer`, False if no
   "answer") / `non_zero_score` (reads **node_chain[1]**, the first step) /
   three last-node variants.
6. **advantage_split**: character-level advantage apportionment + token
   cleanup. Char ledger starts from `ones(len(response_text)) * -777`
   sentinels; the response text is the concat of first node `"\n"+text+`
   (`"\n"` again if multi-node) / middle `text+"\n"` / last `text`, assigned
   per segment by broadcast; token ledger starts from
   `ones(len(input_ids)) * -999` with
   `token advantage = advantage_char[offset_mapping[i][0]]`; asserts
   `tokens[0] == "▁"`; cleanup slices `[1:]` off both.

## I/O

- `score`: `tree, gold, pred_map` → scored tree
- `advantage`: scored tree → advantaged tree
- `paths`: `tree + 4 extraction params` → `[{"node_chain": [...]}]`
- `aggregate`: `variant + paths (+ top_k_percent / keep_at_least_one /
  reduction / k)`
- `filter`: `variant + single path` → bool
- `advantage_split`: `node_chain, response_text, input_ids, offset_mapping,
  tokens` → 4-tuple (all lists)

## Example

A 3-level tree `root → step1 → {2a(correct), 2b → 3b(wrong)}` plus an
early-stopped leaf `shortcut` under root (answer not in pred_map):

- score: 2a=1, 3b=0, 2b=mean(0)=0, step1=mean(1,0)=0.5, shortcut=0,
  root=mean(0.5,0)=0.25
- advantage: step1=0.25, 2a=0.5, 2b=−0.5, 3b=0, shortcut=−0.25
- paths (bf=2, md=3, repeat_early): `[3,3,4,2,2,2,2]` — 2a (depth 2)
  repeats twice, shortcut (depth 1) repeats 4×

## Edge cases and pitfalls

- `repeat_early_stopped_paths` is a misnomer: normally-completed shallow
  leaves repeat too.
- `top_k_percent`'s advantage sequence includes the root (counted as 0), so
  the mean denominator contains it.
- `unique`'s output order is set order; tests must compare as sets.
- `non_zero_score` reads `node_chain[1]`, not the last node.
- Dual sentinels: a response_text longer than the node concat leaves −777
  tails that propagate into the token ledger; multi-char tokens take only
  their start char's advantage.
