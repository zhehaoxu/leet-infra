# Solution Notes

## Approach

The three-stage ledger of the official dvlab-research/Step-DPO data_pipeline (GPT-4o calls stripped):

1. **locate_error_by_gpt4.main** (L26-84): the wrong-prediction filtering ledger — only result-False items go to location; the diagram/asy gate, the strict `>` question2cnt gate, sharding, visited dedup, the `"Step 1: "` header prepending and `\n\n` folding;
2. **prepare_for_correction.main** (L7-73): the GPT-4 verdict parsing ledger — decision segment extraction, the three-stage step_num extraction chain, the `Step {n}:` prefix/continuation split (with the global-replace quirk);
3. **generate_dataset.main** (L6-67): the preference-pair construction ledger — identifier grouping, first-step `"\nStep "` splitting, chosen-first-step dedup, the strip/restore symmetry of the Let's think header.

Out of judging scope: the OpenAI calls and retry loop, the prompt-template assembly (the GPT-4 locator prompt body itself), stepdpo_trainer's loss side (DPO loss on step pairs — boundary with `to-dpo-loss-backward`), and eval_math/evaluation's answer extraction and executor.

Boundary with existing problems:
- **to-dpo-loss-backward / to-dapo-loss** (428 side): DPO's loss and advantage ledger; this problem is the **data side** — how preference pairs are constructed from long CoT (first-error-step truncation);
- **782 to-gmpo-geom-mean**: the on-policy policy-gradient forward ledger; this problem is the offline construction of off-policy preference data;
- **the rl-* batch (rollout/credit)**: online interaction ledgers; this problem is a pure text pipeline with no reward model.

## Key implementation points (pinned to the official source)

- **idx is the pred_data index**: `enumerate(pred_data)` runs after the result filter — the visited filename `{idx}.json` and the shard `idx % n_groups` both use the post-filter numbering, offset from the raw json numbering;
- **gate order is semantics**: diagram filter -> question2cnt gate + increment -> sharding -> visited. The increment before sharding/visited means items skipped by those checks still consume a question's quota (the visited=[3] fixture pins it: idx4 becomes 2>1 skip because idx3 was visited);
- **strict `>` gate**: `question2cnt[question] > max_count_per_question` — a question's 2nd attempt (cnt=1) still passes, the 3rd (cnt=2) skips; check then increment;
- **the `[-1]` chain of the decision segment**: `split("final decision")[-1].split("summary decision:")[-1]` — both headers accepted, the latter wins; the fixture uses "final segment step 9 + summary segment step 2" to pin the summary-override semantics;
- **the step_num extraction chain**: `split("neutral")[0].split("incorrect")[0]` -> `split("\n")[-1]` -> `split(";")[-1]` -> `split("step")[-1].split(":")[0]` -> `int(strip())`. The first two steps compose the cut point "before the earlier keyword" (swapping the split order is equivalent — removed from the mutation list); `count("step") > 1` rejects ambiguous segments like step 2 **and** step 4; decimal "2.5" goes to bad_int (int raises ValueError; a float-parseable variant is not viable);
- **completion_full = prefix + completion**: the prompt's assistant slot holds the already-generated prefix while the completion field holds the continuation **starting from the wrong step** — only their concatenation is the full reasoning (putting the full reasoning in both places doubles it; stepped on in the first draft);
- **the global-replace quirk**: `completion.replace(prefix, "")` — the prefix ends with the marker, so a later repeated marker gets deleted together (`"Step 2: A\nand again Step 2: B"` -> `" A\nand again  B"`);
- **the strip/restore symmetry of Let's think**: prepare products strip the header into the jsonl (identifier alignment), generate_dataset restores it on output (the training prompt needs it);
- **the first-step split `"\nStep "`**: with an intermediate non-`Step `-prefixed line (sub note) the first step **includes** it — unlike `split("\n")[0]` (pinned by the note-line fixture); the rejected side splits the same way;
- **the dedup ledger**: `visited_chosen` resets inside the row loop (per-row dedup), keyed on the chosen first step, and `cnt += 1` sits before the dedup check (deduped attempts count); result-False corrections are filtered at grouping time.

## Test design (10 cases)

collect 3 (basic's sharding-x-gate interaction, gates' five paths including the question2cnt semantics and the max_total cutoff, alpaca style), locate 4 (three decision headers + neutral-only + pass-through fields, four invalid kinds including a decimal step, the global-replace quirk, alpaca style), pair 2 (basic's four pairs with the rejected note line, dedup key/cnt counting/result filtering), args_validation 1 (12 invalid inputs). All expected values were pinned by the official three-script exec sandbox (stubbed openai/jsonlines/tqdm, GPT-4o calls stripped, file IO landing in tmp and read back to verify the {idx}.json naming and content).

## Deviations from the source

- The official locate drops invalids via silent `continue` (four counters); this problem returns `{"valid": False, "reason": ...}` (reasons map one-to-one to the official counters).
- The official three scripts run on the filesystem (glob-read json, jsonlines-write jsonl, {idx}.json on disk); this problem strips them into explicit inputs/returns, and collect's selected corresponds to the officially saved {idx}.json files.
- The official locate main's `while True` API retry loop and its max_count_total cutoff after saving; this problem strips the API loop and keeps the "save first, then cut" break semantics.
- `_split_prompt` is copied three times across the official scripts (two-style if/elif); this problem hoists a common function, matched line by line.
