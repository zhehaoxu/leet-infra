# VinePPO Solution Notes

## Key official details (pinned by sandbox runs)

1. **Pred extraction reads only `node["answer"]`**: `extract_pred_answer`
   calls `task.extract_predicted_answer_from_text(node["answer"])`; the
   node's own `"pred"` field is dead data. Pinned by setting the 2a leaf's
   `pred` to None — the score still follows answer→pred_map. The Shortcut
   leaf's answer `"The answer is 4."` is absent from pred_map → extracted
   None → score 0 despite `"pred": "4"` (the full text is the task's lookup
   key).
2. **`repeat_early_stopped_paths` is a misnomer**: `paths.extend([row] *
   importance_weight)` fires whenever `max_depth > depth` — every leaf
   shallower than max_depth repeats, including the normally-completed 2a
   leaf. For the 3-level tree with bf=2 md=3 the chain-length output is
   `[3,3,4,2,2,2,2]` (2a ×2, shortcut ×4).
3. **Written importance_weight**: `importance_weight if max_depth > depth
   else 1` — nodes at max_depth get 1 (semantically "no exploration budget
   left", coincidentally equal to bf^0).
4. **top_k_percent's mean denominator includes the root**:
   `all_advantages = [node.get("advantage", 0) for node in node_chain]` —
   roots have no advantage, counted as 0 and still divided by.
5. **unique's output order is unreproducible**: a set of json strings,
   subject to PYTHONHASHSEED; tests must compare as sets/lengths.
6. **non_zero_score reads node_chain[1]**: chains are assumed at least
   [root, one step]; the semantics is "did the first step work", not the
   last node.
7. **Dual sentinels −777 (char) / −999 (token)**: `advantage_char` starts
   at `ones(len(response_text)) * -777`; the node-text concat (first node
   `"\n"+text+` optional `"\n"` / middle `text+"\n"` / last `text`) fills it
   exactly when well-formed; a longer response_text leaves −777 tails that
   propagate into the token ledger via `offset_mapping[i][0]` (the −999
   token sentinel only survives when input_ids is longer than
   response_text). Multi-char tokens take only their start char's advantage
   — a token straddling the 0.5/1.0 boundary belongs entirely to the
   pre-boundary segment.
8. **Tokenizer coupling parameterized**: the official
   `out_response.input_ids.reshape(-1)` / `tokens()[0] == "▁"` surface is an
   HF BatchEncoding; here input_ids/offset_mapping/tokens are explicit
   inputs with the `"▁"` assert kept.
9. **extract_paths' side-effect chain**: deepcopy → per-node
   `pop("children", None)` during dfs (official comment: serialization
   speedup) → `json.loads(json.dumps(paths))` round-trip (tuples become
   lists, int/float preserved).

## Boundary with neighboring problems

- `to-gae`: GAE has a learnable value net + TD residuals + λ discount;
  here the value is the MC return mean, no discount, no net.
- `to-ppo-value-clip`: that problem is the value head's clipped loss; here
  there is no learnable value at all.
- `to-grpo-advantage`: GRPO normalizes advantages per group; here it is a
  tree-shaped differencing.

## Sandbox lessons

- When extracting the advantage segment of `convert_path_to_episode`,
  **strip indentation before appending the return** — the return line (0
  indent) participating in the min-indent computation pushes bind to 0,
  leaving the 12-space function body unstripped and exec raising
  IndentationError.
- The Registrable stub needs a **per-base-class registry** (layered via
  `cls.__dict__`), otherwise PathFilter's "none" registration clobbers
  PathAggregator's.
- Exec class bodies directly without indentation prefixes — prefixing
  turns them into nested classes.
- Fixture traps: the multichar-token group originally asserted enc2's
  (all-single-char) output (an out2/out3 mixup) with offsets entirely inside
  the 0.5 segment — fixed to `[30,36)` straddling the boundary plus
  `[34,37)` starting at it; added a −777/−999 sentinel-tail group;
  `paths_repeat_early` is a json.dumps string in the dump and must be
  re-parsed (or compared after json.dumps) in tests.
