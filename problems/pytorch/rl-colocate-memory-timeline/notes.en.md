# In Depth: Colocated Multi-Model Memory and Phase Timeline Accounting

## Origin and background
- verl packs actor/rollout/ref into one `actor_rollout_ref` worker and runs critic as another, moving parameters and optimizer state to the CPU **as whole blocks** between phases via `*_fsdp_config.offload` (`offload_model` / `offload_optimizer`). OpenRLHF's `--colocate` and DeepSpeed-Chat's ZeRO-3 + offload are two spellings of the same mechanism.
- A PPO step's phase sequence is roughly: generation → reward → value → old_log_prob → ref_log_prob → advantage → actor/critic update. **Only the update phases need optimizer state and forward activations**; ref/reward only need to be on-loaded for the one phase in which they are asked something. That is the entire source of "trading transfers for memory".
- This problem writes that schedule as a table and asks for the ledger: peaks, transferred bytes, idle windows, feasibility. Everything is add/multiply over given integers with no real forward (the memory formulas of real forwards live in `training-state-memory` / `distributed-pipeline-1f1b-memory`).

## Four conventions to nail down
1. **All-or-nothing residency**: `resident[p][m]=1` means $\text{param}_m+\text{opt}_m$ is entirely on that card, $0$ means entirely off. There is no half state "parameters present, optimizer absent" — so transferred bytes are always `moves × (param+opt)` and `offload` and `peak` consume the same `full` vector.
2. **Activations follow only train phases and only land on `phase_device`**: this is the line most implementations get wrong. Vectorizing it as `occ += train * act[phase_model]` with shape $(P,1)$ broadcasts into "every card carries a copy of the activation" and inflates `peak`. The activation must be added to the **single column `phase_device[p]` of that phase**. The reference solution deliberately loops here rather than broadcasting, so the convention cannot be violated accidentally.
3. **Closed steady state**: the timeline starts from the phase-0 configuration and must return to it at the end of the step (the next step's generation phase needs to start from the same state). Hence each model's flip count is "differences between phases + the closing put-back", which is always **even** — a free self-check (`check_offload_moves_are_even`). Forgetting the closing term under-counts one transfer that swaps ref/critic back.
4. **Two notions of idle**: `idle_time` is total idleness (how much other work still fits on that card), `max_idle` is the **longest contiguous** idle run (whether a whole extra model or one prefetch fits). In the example GPU1 has 13 total idle but only 8 in its longest run — looking at the total alone overestimates the usable window. No wrap-around across steps, per the statement.

## The feasibility test
`feasible[d] = peak[d] <= hbm_cap` is a per-device inclusive inequality (exactly at the limit counts as feasible, like a real allocator filling a card). A vector is returned instead of a single boolean because colocation plans frequently show "GPU0 overflows while GPU1 has headroom" — reporting only `all(feasible)` would throw away the direction in which to move a model. Searching for a *different* assignment that becomes feasible is out of scope.

## Common pitfalls
- Computing `peak` as $\max_p\sum_d$ (a global peak) rather than per device: the pain of colocation is precisely that **one** card blows up.
- Attributing the activation to the active model's device instead of `phase_device` (see convention 2).
- Forgetting that inference models have `opt=0`: charging optimizer state to ref/reward inflates both `peak` and `offload`.
- With $P=0$, `occ.max(dim=0)` raises on an empty tensor — return explicit zeros, where `feasible` is all True.
- `hbm_cap=0` with any resident model gives all False: this is a table model, not a real allocator with slack.

## Division of labour
- `training-state-memory`: 16 bytes per parameter for one model plus the per-card footprint after ZeRO stage 1/2/3 sharding.
- `distributed-zero-offload`: one model offloading optimizer state/gradients to CPU — GPU memory, CPU memory, PCIe bytes and whether PCIe is the bottleneck.
- `distributed-pipeline-1f1b-memory`: activation residency under 1F1B pipeline parallelism.
- All of those involve a **single model**; this problem is about **several models swapping in and out on the same set of cards according to a phase table**, which introduces transfers/put-back counts, cross-device idle windows and per-device feasibility.
- `rl-rollout-shard-repack` spreads token load across ranks; this one spreads memory across the device-phase plane. The `perf-*` accounting is about time and throughput, this one about capacity and transfers.

## Complexity
$O(P\cdot(M+D))$ to accumulate per phase, plus $O(P\cdot D)$ for the longest idle run. The oracle uses four explicit nested loops with an unfolded state sequence, structurally different from the reference solution for cross-validation.
