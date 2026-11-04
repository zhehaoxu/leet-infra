# Step-DPO: Step-Level Preference Pair Construction (Error-Step Location Parsing + First-Error-Step Truncated Pairing + Filtering Gates)

## Problem

Step-DPO (arXiv:2406.18629, official repo `dvlab-research/Step-DPO`) argues that a long CoT's error usually sits in a single step; running DPO on the whole sequence also punishes the correct later steps. The fix: **truncate at the first error step** and build chosen/rejected pairs only for the wrong step's continuation. This problem takes the three deterministic ledgers of the official `data_pipeline/` (GPT-4o calls stripped; filtering/parsing/pairing all explicit inputs):

```text
mode="collect": items + prompt_style + n_groups/remainder/quotas + visited -> {"selected", "question2cnt", "cnt"}
mode="locate":  gpt4_output + full_prompt + completion + prompt_style -> {"valid", ...}
mode="pair":    corrections + sources + prompt_style -> {"items", "invalid_cnt", "cnt"}
```

## Ledger rules (pinned to the official source)

`mode="collect"` (the filtering ledger of `locate_error_by_gpt4.main`):

```text
pred_data = only items with result False (idx indexes the filtered pred_data)
question contains "diagram" and not "asy" -> skip (Asymptote figures pass)
question2cnt gate: seen and cnt > max_count_per_question -> skip (strict >:
  at most max+1 per question; check then increment; the increment happens
  BEFORE the sharding/visited checks — skipped items still consume quota)
idx % n_groups != remainder -> skip (multi-process sharding)
idx in visited -> skip
selected: completion gets a "Step 1: " prefix; incorrect = prefixed string
  with "\n\n" -> "\n"; folded_solution = same folding on gt_output
save first, then check cnt >= max_count_total -> break
```

Prompt splitting (`_split_prompt`, two styles): qwen2-boxed via
`<|im_start|>user\n` / `\nPlease reason step by step, and put your final answer
within \\boxed{}.<|im_end|>` / `<|im_start|>assistant\n`; alpaca via
`### Instruction:` / `### Response:`.

`mode="locate"` (the per-item parsing ledger of `prepare_for_correction.main`):

```text
correctness = gpt4_output.lower()
              .split("final decision")[-1].split("summary decision:")[-1].strip()
  (both headers accepted; [-1] takes the last — summary wins over final)
no neutral/incorrect -> invalid(no_keyword)
step_num segment = correctness.split("neutral")[0].split("incorrect")[0]
                    then split("\n")[-1].split(";")[-1]
  (neutral-first split order: keeps the text before the earlier keyword)
count("step") > 1 in the segment -> invalid(ambiguous_step)
int() failure of split("step")[-1].split(":")[0] -> invalid(bad_int)
completion_full = prompt's prefix (generated part) + completion (continuation)
no "Step {n}:" -> invalid(no_step_marker) (officially dropped silently)
prefix_out = split("Step {n}:")[0] + "Step {n}:"
output = completion_full.replace(prefix_out, "")
  — global replace: repeated "Step {n}:" segments later also go
returned prefix strips "Let's think step by step.\n"
```

`mode="pair"` (the pairing ledger of `generate_dataset.main`):

```text
corrections grouped by identifier = question + "||" + prefix stripped of
  "Let's think step by step.\n" (official prompts carry the header to strip)
source row identifier mismatch -> invalid_cnt += 1, skip
for each correction of a matched row:
  chosen = the correction continuation; rejected = the source row's output
  first step = split("\nStep ")[0] (before "\nStep ", incl. non-Step lines)
  seen chosen_first_step -> skip that correction (dedup key: chosen first step;
    rejected is NOT deduped)
  cnt increments before the dedup check (deduped attempts included)
output: prefix gets "Let's think step by step.\n" back; original_chosen kept in
  full; dataset/type and answer passed through from the source row
```

## Key quirks (official semantics)

- collect's gate order: the diagram filter sits BEFORE the question2cnt increment (diagram items consume no quota), sharding/visited sit AFTER it (those items do consume quota);
- the per-question gate is `>`, not `>=`: the (max+1)-th attempt for a question still passes;
- locate's `[-1]` segment semantics: when GPT-4 output has a final segment (step 9 verdict) followed by a summary segment (step 2 final verdict), the summary wins;
- `split(a)[0].split(b)[0]` keeps the text before the earlier keyword — neutral and incorrect whichever comes first both cut at the first bad step (swapping the split order is an equivalent mutation);
- output's global replace: a repeated "Step n:" header later in the completion is deleted together (when the target step starts the text and the header reappears);
- pair's dedup only watches the chosen first step and resets per row (visited_chosen is initialized inside the row loop); cnt accumulates across rows;
- the correction prompt's Let's think header is part of the identifier — failing to strip it mismatches everything.

## Input conventions

items/corrections/sources are lists of dicts (missing keys like result raise
KeyError as-is); the four invalid kinds return `{"valid": False, "reason": ...}`
rather than raising (the officially silent skips made observable); the official
`json`/`glob`/`jsonlines` file IO is stripped into explicit inputs.
