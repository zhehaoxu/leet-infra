# PEFT: MultiTask-PromptTuning Shared Pool Concatenation

## Description

**MultiTask Prompt Tuning** (arXiv:2202.13914 / 2211.03831) splits soft prompts into two parts:

1. **Shared prompt** `shared_prompt`: used by all tasks, length \(s\).
2. **Task-specific prompts** `task_prompts`: each task has its own prompt embedding segment of length \(t\).

For a batch, take the task-specific prompt corresponding to `task_ids` for each sample, then concatenate the shared prompt and the task-specific prompt in order to obtain the prefix embedding for each sample:

\[P_b = \bigl[\text{shared_prompt};\ \text{task_prompts}[\text{task_ids}[b]]\bigr],\qquad b=1,\dots,B.\]

The output shape is \((B, s+t, H)\), where \(H\) is the hidden dimension.

## Task

Given:

- `shared_prompt`: float32 tensor of shape \((s, H)\);
- `task_prompts`: float32 tensor of shape \((T, t, H)\), where \(T\) is the number of tasks;
- `task_ids`: int64 tensor of shape \((B,)\), with values in \([0,T)\).

Return the concatenated prefix embedding `P` of shape \((B, s+t, H)\).

- The shared prompt is replicated \(B\) times within the batch.
- Concatenation order is "shared segment first, task segment second".
- Output is float32.

## Hint

- Use `task_prompts[task_ids]` for task-level indexing to get \((B,t,H)\).
- Use `shared_prompt.unsqueeze(0).expand(B, -1, -1)` to expand the shared prompt to the batch dimension.
- Concatenate along the length dimension (dim=1).
