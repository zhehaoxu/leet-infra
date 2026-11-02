# Search-R1 Retrieval Mask Ledger (ag-search-r1-mask-ledger)

**Difficulty**: Hard ｜ **Collection**: agentic-systems ｜ **order**: 730

Turn the multi-turn rollout ledger of Search-R1 (arXiv **2503.09516**) into a pure function: given a batch of prompts, the per-turn generated tokens, the retrieval results and the observation wrapper tokens, recompute each turn's **truncation and action parsing**, the **mask** of the observation span, the two windows — training side (prefix-anchored) and context side (suffix-anchored) — and the rollout statistics. Sourced from [PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1) main: `_postprocess_responses`, `postprocess_predictions`, `execute_predictions`, `run_llm_loop`, `_update_right_side`, `_update_rolling_state`, `_compose_final_output` in `search_r1/llm_agent/generation.py`, plus `search_r1/llm_agent/tensor_helper.py`.

## What you implement

`solve(rows, vocab, search_results=None, wraps=None, max_turns=3, max_start_length=512, max_prompt_length=64, max_obs_length=32, drop_toks=("<|im_end|>", "<|endoftext|>"))`, returning the ledger of the whole batch.

- `rows`: `[{"prompt": [id, ...], "gens": [[id, ...], ...]}]`. `gens` are consumed in order — one per main-loop turn, plus one more for the final rollout if some rows are still active.
- `vocab`: `{id: token text}`, i.e. the tokenizer made explicit (each of `<search>` / `</search>` / `<answer>` / `</answer>` is a single token).
- `search_results`: `{query text: [result token ids]}` (the official code `.strip()`s the result inside `execute_predictions`, so pass the stripped tokens here).
- `wraps`: `{"obs_prefix": [...], "obs_suffix": [...], "retry": [...]}` — the wrapper tokens of `"\n\n<information>" + result + "</information>\n\n"` and the fixed retry prompt for invalid actions.

Returns `{"rows": [...], "active_num_list": [...]}`. Per-row ledger: `actions` / `contents` / `obs_kinds` / `obs_truncated` / `turns` / `valid_actions` / `valid_searches` / `finished` / `right_ids` / `segments` / `dropped` / `final_ids` / `final_info_mask` / `rolling_ids`.

**Only real tokens are booked**: left prompt pads and right compensation pads are always 0 in `attention_mask` / `info_mask`, so they never enter the ledger (proof in notes).

## Ledger points (line-by-line against the source)

1. **Truncation: `</search>` wins, regardless of position.** `_postprocess_responses` is `resp.split('</search>')[0] + '</search>' if '</search>' in resp else resp.split('</answer>')[0] + '</answer>' if '</answer>' in resp else resp` — it first asks whether `</search>` occurs anywhere. So an "answer then search" generation is cut at the **search** close tag, the whole search span stays in the policy tokens (while the action is `answer`); an "search then answer" generation loses everything after the search.
2. **Parsing requires a pair, and is position-first.** `postprocess_predictions` runs `re.search(r'<(search|answer)>(.*?)</\1>', prediction, re.DOTALL)`: the **first opener that has its own close tag** wins, and the content is what sits before its **nearest** closer, then `.strip()`ed. A bare close tag (`</search>` with no `<search>`) yields action `None`; same-kind nesting (`<search>a<search>b</search>`) yields content `"a<search>b"`. Truncation and parsing use different rules, so they can disagree (see point 1).
3. **Special tokens vanish entirely.** After `batch_decode(..., skip_special_tokens=True)` the string is **re-tokenized**: `<|im_end|>` / `<|endoftext|>` do not merely stay out of the policy span, they cost a token slot (one loss bit fewer), and both truncation and parsing run on the **post-drop** sequence (dropping a token in the middle can make two tags adjacent).
4. **An invalid action still burns a turn.** The else branch of `execute_predictions` appends the fixed retry prompt with `dones=0`, `valid_action=0`; the three backslash continuations in the source **do not inject newlines** (the text is joined by single spaces). That turn's policy tokens still count toward the loss.
5. **Observations are truncated from the front.** `next_obs_ids[:, :self.config.max_obs_length]` keeps the **prefix**, so `</information>` is frequently cut away; the retry prompt obeys the same cap (when over-long it is clipped to the same prefix every turn).
6. **The training side is prefix-anchored.** `_update_right_side` computes `effective_len`, then `max_len = min(max_prompt_length, effective_len)`, then `responses[:, :max_len]` — the **oldest** tokens survive. Once full, new policy/observation tokens are **silently dropped**, while the action still takes effect and the row still finishes (`dropped` books the loss). Since `own <= eff`, `min(own, min(P, eff)) = min(own, P)`, so a per-row computation is exact (proof in notes).
7. **The context side is suffix-anchored.** `_update_rolling_state` uses `[:, -max_len:]`, keeping the **newest** window — so the training sequence is "prompt tail + oldest P generated tokens" while the context is "newest P". The usual outcome: the answer reaches the context but never reaches the loss (`rolling_ids != right_ids`).
8. **The prompt is suffix-truncated too.** `initial_input_ids[:, -max_start_length:]` keeps the prompt **tail**; with `max_start_length=0`, `prompt[-0:] == prompt[0:]` is the **whole** prompt, not an empty one.
9. **`max_prompt_length=0` means opposite things on the two sides**: `[:0]` on the right side is empty, `[-0:]` for the context is everything. The same 0 gives two opposite windows.
10. **The final rollout is a "no-retrieval" extra shot.** If rows are still active after the main loop, one more generation runs with `do_search=False`: `next_obs` is discarded entirely (no observation appended) and `turns_stats` is **not** updated; a search there still counts as `valid_action`/`is_search`, but `dones=0`, so the row never counts as finished. Hence `turns` can equal `max_turns + 1` while `finished` stays False.
11. **Statistics**: `turns_stats` starts at 1 and is incremented only for rows that are **still active** after each step, so it equals the number of generations that row actually performed; `valid_action_stats` / `valid_search_stats` accumulate over the main loop **and** the final rollout; `active_num_list` = initial count + after each step + (if any active remain) after the final rollout, and the loop **breaks early** once everyone finishes (no trailing entry).
12. **The `info_mask` convention**: `_compose_final_output` builds `info_mask = attention(prompt) ⊕ attention(responses_with_info_mask)`, where the observation span is replaced by pads on the `_with_info_mask` side, so the mask is 1 for prompt, 1 for policy, 0 for observation. Domain constraints (ValueError): an id missing from `vocab`, a generation missing when actually consumed, a search query with no entry in `search_results`, and any negative bound.

## Example

```python
V = {1: "<search>", 2: "</search>", 3: "<answer>", 4: "</answer>",
     5: "<|im_end|>", 7: "\n\n", 8: "<information>",
     9: "</information>\n\n", 11: "RETRY", 20: "Q",
     30: "weather", 31: "42", 40: "Doc1", 41: " Doc2"}
W = {"obs_prefix": [7, 8], "obs_suffix": [9], "retry": [11]}
rows = [{"prompt": [20], "gens": [[1, 30, 2], [3, 31, 4]]}]
led = solve(rows, V, {"weather": [40, 41]}, W, max_turns=3)
r = led["rows"][0]
# actions == ["search", "answer"]; contents == ["weather", "42"]
# turns == 2; valid_searches == 1; finished == True
# segments == [["policy", 3], ["obs", 5], ["policy", 3]]
# final_info_mask == [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1]
# active_num_list == [1, 1, 0] (finished on turn 2 -> break, no final rollout)
```

## Scoring

15 checks: search-first truncation, paired parsing with the nearest closer, the retry turn of an invalid action, prefix truncation of observations (retry prompt included), the three semantics of the final rollout, the training-side cap with `dropped`, the divergence of the two windows, prompt tail truncation and the meaning of `0` in a slice, the two-faced `max_prompt_length=0`, special-token dropping, the shape of `active_num_list`, the four defaults, domain constraints, and 60 randomized rollouts compared against the oracle.
