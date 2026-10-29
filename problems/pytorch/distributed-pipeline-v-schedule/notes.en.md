## Reference approach

Map chunks to physical devices using alternating sweeps. Maintain forward and backward completion counts per chunk. Because micro-batches within a chunk are prioritized by ascending ID, each count identifies that chunk's next candidate micro-batch.

Forward candidate $b$ requires the previous chunk's forward count to exceed $b$. Backward candidate $b$ requires its own forward count and the next chunk's backward count to exceed $b$; omit the downstream condition at the terminal chunk.

First select tasks for every device, then commit all completions. Updating counts immediately after selecting one device would let subsequently visited devices consume results within the same slot, making traversal order affect the timeline and violating dependency timing.

## Measure peaks instead of guessing them

Increase live activations at forward completion and decrease them at backward completion; record each device's high-water mark. An activation remains live during backward, already counted at the preceding boundary. With at most one task per device per slot, no same-device release/allocation ordering ambiguity exists.

At the end of all forward work, the flush baseline holds every activation: $mv$ per device. The $\min(m,p-i)$ bound of a different 1F1B schedule does not apply, and there is no universal $\lceil(p-i)/v\rceil$ savings formula. More virtual chunks do not guarantee monotonic time or memory improvement under this policy. Each virtual task has fixed duration 1 here; comparing different $v$ is not a fixed-per-layer-duration comparison.

## Verification and complexity

Tests independently build the explicit task DAG, track indegrees and schedule per-device priority queues, unlike the reference's completion counters. A separate replay checks dependencies, placement, exactly $2mpv$ tasks, zero final live activations, and agreement between peaks and busy cells. A single device executes serially without bubbles; one micro-batch still obeys folded same-device dependencies; zero activation bytes changes only memory, not scheduling.

The reference checks $pv$ chunks per slot: $O(Tpv)$ time, $O(pT)$ output storage, and $O(pv)$ auxiliary state. The finite acyclic task graph always permits progress until completion.

## Source and simplifications

[Zero Bubble Pipeline Parallelism](https://arxiv.org/html/2401.10241v1) discusses V-shaped placement and separating input-gradient from weight-gradient work. This exercise borrows only the placement and explicitly defines its own ready-task priority. It does not reproduce ZBV scheduling tables, weight-gradient bubble filling, communication overlap or optimizer synchronization, and does not equate V placement with Megatron interleaved 1F1B.
