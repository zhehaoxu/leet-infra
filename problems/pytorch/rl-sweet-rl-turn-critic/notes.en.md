# SWEET-RL Implementation Notes (792)

Source: facebookresearch/sweet_rl `scripts/` (four scripts) + paper arXiv:2503.15478 §"SWEET-RL". Training lives in the companion fork YifeiZhou02/collab_openrlhf (an openrlhf fork with --mean_log_prob supporting length-normalized DPO). Sandbox oracle: a faithful re-execution of the official logic + a pair-by-pair cross-check under `random.Random(42)` (/tmp/sweet_check.py).

## Per-entry semantics

### judge_prompt (rank_best_of_n.py::formatting + main prompt section)

- GT branch: `"In light that the final answer is: {ground_truth}." + input + output + RESPONSE_TEMPLATE`; the no_use_ground_truth branch drops the prefix.
- RESPONSE_TEMPLATE = `"<|start_header_id|>assistant<|end_header_id|>\n\n"` (the Llama-3.1 chat template). **No space** between the prefix and the input (direct f-string concatenation).

### judge_score (rank_best_of_n.py::main scoring section)

Four steps, order pinned:
1. `prompt_logprobs[:-4]` drops the tail — official comment 'the last four digits are another empty response'.
2. Inside the loop `if prompt_logprob is not None:` gates both appends — the logprob and decoded_token are collected **in sync** (a None position contributes to neither).
3. `find_last_index(all_tokens, eos_token)` — the **last** eos (`len(lst)-1-lst[::-1].index(element)`, -1 when absent).
4. `tail = logprobs[last+1:]`; use_sum → `sum(tail)`, else `sum(tail)/len(tail)` (mean = length normalization, the default).
- Each prompt_logprob entry is a `{token_id: Logprob}` dict (one entry under prompt_logprobs=0 sampling); the official code iterates `.values()` (this problem parameterizes entries as (logprob, token) tuples).
- An empty tail raises ZeroDivisionError — real data guarantees at least 4 truncated tail positions after the eos (see the judge_score fixture design).

### rank_against_reference (rank_best_of_n.py::main rerank section)

- `judge_results - reference_judge_results`, both reshaped (-1, best_of_n); the reference comes from sample_best_of_n's additional_reference_logprobs computed with the **old policy** (mean pairs with the judge mean, sum with sum).
- `torch.argsort(descending=True, dim=1)` → this implementation `np.argsort(-adjusted, axis=1)` (identical stable order: equal scores by ascending index, pinned by the [0,1,3,2] anchor).
- The official code writes judge_result back into data for generate_random_pairs to consume.

### flatten (sample_best_of_n.py::main flattening section)

- `for i, dh in enumerate(dialogue_history): if "input" in dh:` — only entries carrying "input" expand (the human simulator's utterances; the agent's output-only entries skip).
- Each record {input, input_with_ground_truth (GT prefix + input), older_output, reward, ground_truth, messages[:i+1]} — messages truncated to the current turn; this problem keeps input/older_output/messages plus input_with_ground_truth.

### pairs (generate_random_pairs_from_ranks.py::main)

- `random.seed(42)` is a **script-level global** seed; this implementation takes `random.Random(seed)` — seed=42 reproduces the official draw sequence exactly (sandbox pair-by-pair: choice takes chosen first then rejected, then the shuffle).
- `extract_indices`: `np.argsort(judge_results)` **ascending**; `chosen_index = random.choice(range(n//2))`, `rejected_index = random.choice(range(n//2, n))` — selected **by position**, returning `argsorted_indices[position]` (the sorted original index). The main loop then swaps by the **original score** so chosen > rejected — net effect: chosen is the higher scorer of "a lower-half pick" vs "an upper-half pick", **usually not the absolute best** (only the bestworst variant takes the argsort extremes).
- The draws repeat num_pairs times over the same judge_result (rng state carries across).
- get_sft: after `random.sample(range(n), 2)`, **chosen=rejected=indices[0]** (the official quirk — SFT needs only positives); the text is always output+eos with a prompt.
- no_prompt: `input + output + eos` (the context rides in the response side); otherwise `output + eos` with a separate prompt.
- Finally `rng.shuffle(preference_data)`.

### gt_pairs (evaluate_code.py::main preference section)

- The flat k-repeat list reshapes to (k, -1); repeat j of task i sits at `j*num_tasks + i`.
- `best_repeat = argmax(arr[:, i])`; for every `j != best_repeat and arr[j,i] < arr[best_repeat,i]` one pair {chosen_index, rejected_index} — **strictly <** (a == max non-argmax repeat produces none).
- Text assembly (this implementation returns indices; the text ledger is pinned here): the GT variant's chosen/rejected = GT prefix + `dialogue_history[-1]`'s input+output (the **last turn**); the non-GT variant has no prefix (and the official code assigns the chosen variable twice to the same value — no behavioral difference).

## Sandbox and tests

- /tmp/sweet_check.py: a direct re-execution of the official logic (extract_indices' argsort-position semantics, the judge four-step extraction, the pairs main loop + Random(42)) cross-checked against the solution pair-by-pair; judge_score hand anchors (PL: tail=[-0.4,-0.5,-0.6,-20,-21,-22] → mean=-10.75, sum=-64.5; PL_SHORT: tail=[-0.4]).
- Fixture lesson: the judge_score fixtures must count the interaction of the [:-4] truncation with the EOS position — the first two drafts got the truncation boundary wrong (tail mixing in / missing x tokens); pin the boundary with a short case ("exactly 1 real token after the EOS + 4 truncated") before the long one.
- gt_pairs reshape direction: the flat list is [j0i0, j0i1, j1i0, ...] — draft the reshape before asserting (the first draft treated [[1,1,1],[0,0,0]] as the rows of reshape(3,2), backwards).

## Mutation checklist (mut792 pending)

Easy-to-miss spots: the [:-4] count, the None skip, find_last_index last vs first, the tail slice +1, the use_sum branch, the GT prefix concatenation, RESPONSE_TEMPLATE, subtracting vs not subtracting the baseline, the descending argsort, the ascending-argsort position semantics (the n//2 half boundary), the swap condition > (strict), get_sft's chosen==rejected, the no_prompt concatenation order, the shuffle, and gt_pairs' argmax / strict < / index formula.

## Boundaries

- 789 `rl-tournament-grpo`: that judge compares pairwise and picks winners (win counts); this critic scores absolutely (logprob-derived) minus a reference baseline and pairs by ranking.
- 790 `rl-opsd-turn-credit`: OPSD's teacher shares the student's snapshot and multiplies a belief-recursion credit; SWEET-RL's critic is a separately trained privileged scorer whose product is DPO preference pairs.
- The to-dpo family: the DPO loss itself is already taken; this problem is the data-construction side (critic scoring → ranking → pairing).
