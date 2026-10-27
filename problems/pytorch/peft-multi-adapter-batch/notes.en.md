# PEFT: Multi-Adapter In-Batch Hot-Swap Routing — Algorithm Notes

## 1. Provenance and system context

S-LoRA (Sheng et al., OSDI 2024, arXiv:2311.03285) and punica (Chen et al., arXiv:2310.18547) established the "one base, thousands of adapters" serving form: LoRA weights are under 1% of the base parameters, thousands of adapters can stay resident in memory, mounted dynamically per request. The decode-phase bottleneck is the GEMV with "different adapters for different requests in one batch" — punica fuses the id-segmented gather-matvec with a custom **SGMV** kernel, S-LoRA manages memory with unified paging. This collection's #79⑦ already covers the memory/LRU account; this problem covers only the **forward composition**.

## 2. Decomposition of the forward (judged points)

\[y=\underbrace{xW_0^{\!\top}}_{\text{shared base}}+\underbrace{\sum_k \mathbb 1[\text{ids}=k]\,s_k\,(xA_k^{\!\top})B_k^{\!\top}}_{\text{row-routed increment}}\]

The implementation is equivalent to "group by k, batched GEMV within the group, scatter back by row"; four semantics are pinned:

- `check_zero_scales_identity`: all \(s_k=0\) → exactly \(xW_0^\top\) (increments fully off, base shared).
- `check_single_adapter_consistency`: ids all 0 → \(y=xW_0^\top+s_0(xA_0^\top)B_0^\top\), bit-for-bit identical to the single-adapter forward (routing degenerates).
- `check_scale_doubling_scope`: after \(s_1\to 2s_1\) **only the rows with ids==1** change — routing is row-isolated; any cross-talk is a bug.
- The scale \(s_k\) is the interface for merging/weighting multiple adapters (e.g. adapter mixing), not decoration — the example and the random cross-check both make it non-1.

## 3. Division of labor within the collection

- `peft-lora-forward`: single-adapter forward; `peft-adalora-budget`: training-time rank allocation.
- Serving memory / swap-in-swap-out (LRU, cold start) belongs to #79⑦.
- This problem: in-batch routing + the **forward semantics** of scaled composition.

## 4. Verification approach

`check_example` hand computation (\(r=1,K=2,N=3\), \(W_0\) the identity): samples 0/2 attach adapter 0 (the scaled outer product of \(2\cdot[1,0]^{\!\top}\)), sample 1 attaches adapter 1 → \(y=[[3,0],[0,4],[3,1]]\); `check_matches_oracle` compares against a per-sample per-scalar recomputation (float64 internally, float32 out, tolerance \(10^{-5}\)).
