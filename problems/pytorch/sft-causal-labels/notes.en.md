# Explained: Causal LM Shifted Labels

## Origin
The standard GPT-style autoregressive objective (Radford et al. 2018; HF `CausalLM` internally does `shift_labels = labels[:, 1:]` and aligns it with `logits[:, :-1]`, using `ignore_index=-100`). This function makes "shift left + mask" explicit and is the canonical way to write `label = next_token if supervised else -100` in an SFT pipeline.

## Core Idea and Pitfalls
1. **Check the successor's mask**: `labels[b,i]` corresponds to "predict \(i+1\)", so **the decision uses `supervised_mask[b, i+1]`** (whether the next token is a supervised target), not `i` itself. Writing `mask[i]` is the most common misalignment.
2. **Last column always -100**: \(i=L-1\) has no in-row successor and must be ignored; you cannot take `input_ids[b+1,0]` from the next batch row as the target (cross-sample leakage).
3. **`ignore_index` is configurable**: defaults to `-100`, filled per the argument; do not substitute 0 or pad, which would clash with real tokens.
4. **Source of the value**: supervised positions get `input_ids[b, i+1]` (the predicted token), not `i`.

## Cross-checking Details
The reference solution uses "doubly nested loops, `if mask[b,i+1]>0: labels[b,i]=ids[b,i+1]`, keeping the last column ignored". The judge's `_oracle` builds labels independently with **per-row python lists**: copy the whole row as `-100`, then for each row's `1:` slice select and back-fill `ids[1:]` according to `mask[1:]` (equivalent to "shift first, then select by successor mask"), a different structure from the reference solution but elementwise identical. It also asserts "all-zero mask → all ignored", "all-one mask → pure shift with last column ignored", the "example", and "multi-row last column never crosses rows".

## Complexity
\(O(B L)\).
