# In Depth: The Byte Ledger of Gradient Compression, the Fallback Switch and the Maximum Feasible Bit Width

## Source

- **1-bit Adam / sign compression**: Li et al., *1-bit Adam: Communication Efficient Large-Scale Training of Adam's
  At Scale* (ICML 2021, arXiv:2102.02895; DeepSpeed's `1bit-adam`). Real implementations attach one fp32 scale to a
  group of elements (e.g. 148x1024 of them) besides the sign bits; here that granularity is the explicit
  `scale_group_size` parameter.
- **PowerSGD**: Vogels, Karimireddy, Jaggi, *PowerSGD* (ICML 2019) - each gradient matrix becomes a rank-$r$ product
  of $P\in\mathbb{R}^{rows\times r}$ and $Q\in\mathbb{R}^{cols\times r}$, so the volume drops from $rows\cdot cols$
  to $(rows+cols)\cdot r$.
- **Error feedback**: Karimi et al., *Error Feedback Fixes Signum-gradient and Other Gradient Compression Schemes*
  (MLSys 2019) - the dropped residual has to be accumulated locally in fp32, so compression **saves bandwidth, not
  memory**.
- The ring all-reduce coefficient $c=(W-1)/W$ with $2cS$ bytes per rank is reused from this bank's
  `distributed-zero-communication`.

## Key Points and Pitfalls

1. **Bit packing rounds up**: $\lceil numels\cdot bits/8\rceil$ is an integer ceiling, not floating-point division.
   With `bits=3` and $numels=100$ it is $\lceil 37.5\rceil = 38$ bytes; truncating with `int(...)` undercounts every
   case that is not a multiple of 8.
2. **`scale_group_size=0` means "one scale shared by the whole tensor", not "no scale"**: 4 bytes per group, and
   **each matrix pays its own**. With $g=1$ the scale cost alone is $numels\cdot 4$, already the full fp32 volume, so
   adding sign bits always triggers a fallback; $g=numels$ is exactly equivalent to $g=0$, which is the degenerate
   point the grader uses to pin the convention.
3. **Low rank is a linear cost with a full-rank cap**: the payload is $(rows+cols)\cdot r\cdot bpe$ (two factors
   added - not $r^2$, not $r\cdot\min(rows,cols)$). When $r\ge\min(rows,cols)$ the matrix is treated as **full rank**
   and shipped as is. At $r=\min-1$ the product is usually **larger** than the original: $1024\times1024$ with
   $r=1023$ gives 8,379,904 bytes versus 4,194,304 for fp32 - compression that costs more, so it must fall back.
4. **The fallback test is strict**: at `payload == full_payload` we get `compressed == full`, `fallback=False`,
   `saved_ratio=0.0`; writing `>=` would produce a spurious True for exactly-full-rank matrices or for
   `method="full"`.
5. **The ring coefficient $c=(W-1)/W$**: `per_gpu_comm_bytes = 2cS`, which is $0$ at $W=1$ (no all-reduce needed) and
   approaches $2S$ only as $W\to\infty$. Doubling $W$ does not double the traffic ($1.5\to1.75$). `saved_ratio` is a
   ratio of payloads and is independent of $W$ - do not fold $2c$ into it.
6. **Error feedback is always fp32**: `ef_state_bytes = total_elements*4`, independent of `bytes_per_element`,
   `method` and bit width; only `error_feedback=False` makes it 0. The memory bottleneck of 1-bit Adam was never on
   the communication side.
7. **`max_bits` is the largest bit width that still does not fall back**: the payload is monotone non-decreasing in
   the bit width, so a forward scan keeping the last feasible value is enough (binary search only invites off-by-one
   bugs on tiny tensors). It is often `bytes_per_element*8 - 1`, because those 4 scale bytes eat exactly the last
   margin; non-sign methods report 0, and so does `full_payload == 0` (empty shape list or all-zero sizes).
8. **Empty matrices are skipped entirely**: at $numels=0$ neither packing nor scale applies; charging "at least 4
   bytes per matrix" would add 8 phantom bytes for `shapes=[[0,64],[64,0]]`.
9. **Division of labour**: uncompressed volumes and shard ownership live in `distributed-zero-communication` /
   `distributed-zero-shard-ownership`, parameter/optimizer **bytes** in `training-state-memory` /
   `transformer-mem`, HBM traffic in `operator-fusion`; this problem is only about the **bytes** shipped across
   cards after compression and the switch points.

## Cross-check Approach

The reference solution ceilings with `(a+b-1)//b`, writes low rank as `(rows+cols)*rank*bpe`, the ring coefficient as
`2.0*(W-1)/W`, and scans `max_bits` **forward from 1**. The grader's `_oracle` takes different routes everywhere: the
full payload accumulates in 8-row blocks (`min(8, rows-start)*cols*bpe`); packing uses the remainder of
`divmod(n*bits, 8)` to decide the carry; the group count is `(n-1)//g + 1`; low rank becomes two products
`rows*rank*bpe + cols*rank*bpe`; the ring coefficient becomes `payload*(2 - 2/W)`; `saved_ratio` uses the integer
difference `(full-compressed)/full`; and `max_bits` scans **downward from the highest bit** for the first feasible
value (opposite direction, same answer). Structural assertions cover: zero traffic at $W=1$, coefficients
$1/1.5/1.75$ for $W\in\{2,4,8\}$, guaranteed fallback with `max_bits=0` at $g=1$, $r=\min$ not falling back while
$r=\min-1$ does, the invariants `compressed <= full` and `per_gpu <= full_comm` over a 72-point grid, the all-zero
empty shape list, and `error_feedback=False` changing only item 6.

## Complexity

The reference solution is $O(K + K\cdot B)$ with $K$ the number of matrices and $B=$ `bytes_per_element*8` (the
linear `max_bits` scan). The oracle's block loop is $O(\sum rows/8)$; the whole suite finishes in tens of
milliseconds.
