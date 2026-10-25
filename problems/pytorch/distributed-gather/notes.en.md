# Approach

## Interviewer's Intent

The distinction between the two pairs "single destination vs all destinations" — Gather / All-Gather and Scatter / Broadcast — is where collective communication is most often glossed over. Answering immediately that "in Gather only one GPU gets the whole, the others stay unchanged" shows that you truly understand the **destination** semantics of communication instead of treating `all_*` as the default.

## Semantics

Every rank hands over one shard of equal length; the concatenated complete result is **delivered only to `dst`**. The destination rank's buffer becomes $\bigoplus_q \texttt{inputs}[q]$, and **non-destination ranks keep their own shard unchanged** (gather gives them no result). This problem takes this MPI-undefined behavior for non-destination ranks and **stipulates that they keep their input as-is**, making grading deterministic.

## Comparison with Neighboring Primitives (key point)

- **All-Gather**: the concatenated result goes to **all** ranks (the "broadcast version" of Gather).
- **Gather**: the concatenated result goes only to `dst`.
- **Reduce** (for contrast): the data of all ranks is **summed** and given only to `dst` — aggregation rather than concatenation. This problem is numerical concatenation with no addition.

## Implementation Notes

`torch.cat(inputs, 0)` yields the complete result, which goes into `out[dst]`; for the rest, `out[j]=inputs[j].clone()`. Do not fill the non-destination ranks with the complete result too — that would turn it into All-Gather (`check_only_dst_has_full` is dedicated to catching this). `dst` is not necessarily 0 (`check_various_dst` iterates over it). Use `.clone()` to ensure input storage is not shared.

## Real-world Usage

`torch.distributed.gather` / NCCL `gather` is often used to aggregate per-GPU local metrics (loss, grad-norm, token counts) onto rank0 for logging without sending the whole back to everyone; it saves a broadcast compared with All-Gather, and thus one extra copy of traffic. It is the inverse of `distributed-scatter`: scatter cuts `dst`'s complete data into one shard per rank, while gather collects one shard per rank back into the complete data at `dst`.

## Grading Approach

`test_cases.py`: the sample, only dst gets the whole while the rest keep their input, iterating `dst` over values other than 0, a single rank, and no modification of inputs; the independent reference concatenates manually element by element (not via `torch.cat`) for cross-checking.
