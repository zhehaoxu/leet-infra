# SFT Data: Assistant-Reply Loss Mask

## Problem

A golden rule of supervised fine-tuning (SFT): **only compute the loss on assistant-reply tokens**; system / user / role separators / padding must not be supervised — otherwise the model learns to "recite" what the user said. This is TRL's prompt-loss masking, the assistant-only loss mask used by various frameworks. This problem reconstructs that 0/1 supervision mask from the per-token role labels.

## Rules

Implement `solve(token_ids, role_ids, assistant_id, eos_id, supervise_eos=True)`, returning a float32 mask of shape \((B, L)\) with values in \(\{0.0, 1.0\}\).

- `token_ids`, `role_ids`: int64 \((B, L)\), already right-padded to equal length \(L\) within the batch (the role at pad positions is not equal to `assistant_id`). `role_ids[b, i]` is the role index of the \(i\)-th token of the \(b\)-th row.
- Mark the **contiguous runs** of positions whose role equals `assistant_id` (a single maximal assistant run) as supervised, i.e. the mask takes 1.
- When `supervise_eos=False`, for each assistant run drop **only** its **last** token (if at that position `token_ids[b, ·] == eos_id`); an `eos_id` appearing in the middle of a run is kept. With `supervise_eos=True` (default) the whole run is supervised.
- All other positions (non-assistant, dropped run-final EOS, pad) take 0 in the mask.

## Examples

`assistant_id=2, eos_id=99, supervise_eos=False`:

```
token_ids = [[5,6,7,8,99,0],      role_ids = [[0,1,2,2,2,3],
             [1,2,99,3,4,0]]                  [1,2,2,2,3,3]]
mask      = [[0,0,1,1,0,0],       # row 0 assistant run [2,3,4]; its last token tok=99=eos is dropped
             [0,1,1,1,0,0]]       # row 1 assistant run [1,2,3]; last tok=3≠99 so the whole run is kept (a 99 in the middle of a run is never dropped)
```

For the same input with `supervise_eos=True`, row 0's mask becomes `[0,0,1,1,1,0]`.

## Constraints

- CPU int64 input, float32 output, shape strictly \((B, L)\).
- An assistant run is a maximal contiguous run of "role equals `assistant_id`"; different runs do not affect each other.
- `supervise_eos` only affects the **last position** of each run, and only when the token at that position equals `eos_id`.
- Output values are only 0.0 / 1.0.
