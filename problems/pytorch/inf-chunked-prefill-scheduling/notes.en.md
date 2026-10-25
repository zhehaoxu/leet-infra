# Solution Notes · Chunked Prefill

## Why chunking is needed

prefill is compute-bound (the whole prompt is computed at once), while decode is memory-bandwidth-bound (1 token per step). Letting a single long prefill monopolize a step inflates the inter-token latency (ITL) of the decode requests in the same batch. Chunked prefill breaks the long prefill into "a small bite per step" and runs it mixed into the same batch as decoding, making ITL more stable while also filling the GPU's compute budget.

## Budget allocation order

The problem defines the priority as "decode takes first, the remainder goes to prefill": $\text{avail}=B-n_\text{dec}$. This corresponds to vLLM's policy where decode tokens are always scheduled and prefill uses the leftover. When $n_\text{dec}\ge B$ (decoding has filled the budget), this prompt cannot enter at all in this step; `avail<=0` returns an empty list directly — the easiest boundary to overlook.

## Difference from whole-model scheduling

In a real scheduler, $n_\text{dec}$ at each step rises and falls as requests complete, and multiple waiting prefills compete for the same `avail`. This problem pins down only "the chunk sequence of a single prompt under constant decode occupancy", stripping out the dynamic part to keep grading deterministic and the formula pure. The chunk size depends only on $\text{avail}$, and the elements always sum to `prompt_len`.
