# LLM Inference Throughput in Depth

## Interviewer's intent

"Why does TTFT grow as the prompt gets longer, while the cost of each additional generated token stays essentially constant?" "Why does batching decode
costs almost nothing in latency but does not add throughput?"—all these core serving phenomena can be explained clearly from the
FLOPs structure of the prefill/decode phases. This problem turns it into a computable account using the $2N$ convention.

## $2N$ per token

In one forward pass, each parameter participates in one "multiply + add", so each token costs about $2N$ floating-point operations. This is the counterpart, for a
**pure inference forward pass**, of the $6ND$ training convention (forward $2ND$ + backward $4ND$) in `transformer-mfu`: here we take only $2N$.

## The essential difference between the two phases

- **prefill**: `prompt_len` tokens are computed **all at once in parallel**, the matrices are large → high arithmetic intensity → **compute-bound**.
  FLOPs $=\text{batch}\cdot\text{prompt\_len}\cdot 2N$, which determines TTFT.
- **decode**: each step has only **1 new token** (parallel along the batch dim but sequence dim = 1), the matrices are thin and long → each read of the weights
  only does $2N\cdot\text{batch}$ of work → often **memory-bound** (weights + KV cache must be re-read).

This problem converts the time of both phases with the same `effective_flops`, which is a simplified teaching convention; in real systems decode is often limited by
bandwidth rather than compute (see `roofline-arithmetic` and `kv-cache-memory`).

## Optimization directions this accounting reveals

- To reduce TTFT: prefill is a compute job → increase parallelism / interleave chunked prefill with decode.
- To increase throughput: **continuous batching** packs the decode of multiple requests into a large batch, amortizing the weight reads per token
  and pushing memory-bound decode toward compute-bound.
- `gen_len` only affects the decode term and does not affect TTFT—this is exactly the structure pinned down by `check_zero_generation`.

## Common mistakes

- Counting prefill as growing with `gen_len` (prefill consumes the prompt only once).
- Writing $2N$ as $N$ or $4N$ (that is the training forward+backward convention).
- Using integer division and truncating the time (convert to float first).

## Grading criteria

`test_cases.py` recomputes FLOPs and time independently per phase to cross-check, and pins down: `gen_len=0 → no decode`, the linearity of batch in
FLOPs/time, doubling `effective_flops` only halves the time without changing FLOPs, and the invariant `total ≥ ttft`.
