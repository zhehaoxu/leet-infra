# PEFT: MultiTask-PromptTuning Shared Pool Concatenation — Algorithm Notes

## 1. Formula and Source

MultiTask Prompt Tuning comes from arXiv:2202.13914 (MPT) and arXiv:2211.03831 (task-routed extension of MPT). It splits soft prompts into:

- `shared_prompt`, shared across all tasks;
- `task_prompts[task_id]`, unique to each task.

For a batch, the prefix embedding for each sample is:

\[P_b = \bigl[\text{shared_prompt};\ \text{task_prompts}[\text{task_ids}[b]]\bigr].\]

## 2. Implementation Details

- `task_prompts[task_ids]` uses advanced indexing to fetch \((B,t,H)\) in one go.
- `shared_prompt.unsqueeze(0).expand(B, -1, -1)` replicates the shared segment to the batch dimension.
- Concatenate along dim=1 (the sequence-length dimension) to obtain \((B, s+t, H)\).

## 3. Degenerate Cases and Grading Points

- Batch size \(B=1\): shared segment must still be correctly replicated and concatenated.
- All samples share the same `task_id`: shared segment is replicated per sample, task segment is identical.
- Concatenation order must be "shared first, task second", verified by an inverted-order counterexample.
- Fully deterministic, no randomness.

## 4. Split with Existing Problems

- `peft-prompt-tuning`: single-task learnable continuous prompts.
- `peft-prefix-tuning`: Prefix-Tuning prefix generation.
- This problem: shared pool + task-specific segment concatenation in a multi-task setting.
