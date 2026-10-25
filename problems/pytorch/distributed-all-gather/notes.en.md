# Approach

## Interviewer's Intent

Collective communication primitives are the "alphabet" of distributed training. When an interviewer asks "what is the difference between All-Gather and All-Reduce" or "which step in tensor parallelism is an all-gather", failing to answer shows that you can only call APIs and do not understand how data flows between GPUs. This problem breaks the most basic primitive down into a pure function for you to implement; `reduce-scatter` / `all-reduce` then assemble the complete set.

## Semantics

$W$ ranks each hold one shard of equal length; All-Gather **concatenates all shards in rank order**, and **every rank receives the same complete result**. Modeled as a pure function: the input `shards` (a list of length $W$), and the output is also a list of length $W$ whose every item is `torch.cat(shards, dim=0)`.

## Comparison with Other Primitives (running through this collection)

| Primitive | Input (per rank) | Output (per rank) | Same result on every rank |
|------|----------------|----------------|----------------------|
| All-Gather | one shard $S/W$ | full $S$ (concatenated) | Yes |
| Reduce-Scatter | full $S$ | **one shard** after reduction | No (each gets a different piece) |
| All-Reduce | full $S$ | **full $S$** after reduction | Yes |

All-Gather **performs no reduction** (no adding or summing); it only moves data together — so its output volume is $W$ times its input.

## Real-world Usage

- **Tensor parallelism (Megatron)**: the output of a column-parallel Linear is "one column block per GPU"; a single All-Gather stitches them into the full activation.
- **Inference TP**: partial logits computed on each GPU are merged into the complete sequence via All-Gather.

## Common Mistakes

- Concatenating in an order other than rank order (using a set / shuffling), which puts `full` in the wrong order.
- Mistakenly performing a sum (that is All-Reduce).
- Returning a single tensor instead of a "one per rank" list — the contract here requires a list of length $W$.

## Grading Approach

`test_cases.py` uses `torch.cat(shards,0)` as the reference and verifies: every rank's result is identical and in rank order, single-rank degeneration, 2-D shards concatenated along dim0, and that the **input is not modified**.
