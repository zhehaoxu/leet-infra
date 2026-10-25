# Solution Notes · MoE Expert-Parallel All-to-All

## Why all-to-all

In TP, each device holds a slice of the experts for all tokens; EP is the opposite — each device holds a slice of all experts plus a slice of the tokens. Once the routing decisions are made on the local device, tokens must be sent **across devices** to the device hosting the target expert; that single communication is called all-to-all: every device sends to every device.

## Semantics of send_matrix

`send_matrix[i, j]` = the number of (token, slot) pairs sent from device i to device j. It is **not the token count**: if a token's top-k selects two experts on different devices j1/j2, it is counted once each in `send_matrix[i, j1]` and `send_matrix[i, j2]`.

## Combine weights

Both the original paper and mainstream implementations (GShard/Megatron/VLLM) require `topk_weights` to sum to 1 per row (**already renormalized during routing**), so `combined[t]` is the weighted average of the expert outputs. The statement stipulates that `topk_weights` is already renormalized, so `solve` does no second normalization.

## tokens_per_device uses ceil

`tokens_per_device = ceil(T / P)` guarantees that the first devices are filled and the last one may hold fewer. `src_device(t) = t // tokens_per_device` is a contiguous sharding. In the statement's example T=8, P=2 → tpd=4 divides evenly.

## Relation to distributed-all-to-all

`distributed-all-to-all` is the **general** version of the collective-communication primitive (irregular sharding, supports multiple rounds); this problem is specific to MoE: the sharding rule is determined by (num_experts, experts_per_device) and (T, k), which makes it a typical all-to-all workload.
