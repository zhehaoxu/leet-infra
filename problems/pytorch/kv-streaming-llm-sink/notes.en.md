# Solution Notes · StreamingLLM

## Why score it this way

Every `ScorerPress` in kvpress relies on `scores.topk(n_kept, -1).indices` to select the positions to keep. StreamingLLM has no notion of continuous importance; it only needs "the middle gets evicted first"—a 0/1 binary score is enough, since `topk` will pick the 1s first; the order in which the remaining positions are picked from the 0s does not matter (they all get pruned anyway).

## Rounding of n_pruned

`n_pruned = k_len - int(k_len * (1 - compression_ratio))`. It uses **int** (truncation) rather than round, consistent with kvpress's `compute_n_kept`: `n_kept = int(k_len * (1 - cr))`, `n_pruned = k_len - n_kept`.

Example: `k_len=10, cr=0.4` → `n_kept=int(6.0)=6` → `n_pruned=4`; `k_len=8, cr=0.25` → `n_kept=int(6.0)=6` → `n_pruned=2`.

## Intuition behind the sink

Softmax requires "the denominator not to be 0"; under a causal mask the first token can only see itself, and the model learns to dump "excess attention" onto the earliest position. If the sink is pruned, attention goes out of control. The sink count is therefore usually 4; here it is passed explicitly as a parameter to make cross-checking easy.

## Role in the full pipeline

`StreamingLLMPress` is a pure positional rule that ignores content—the cheapest option, and the number-one control baseline for all KV-compression methods.
