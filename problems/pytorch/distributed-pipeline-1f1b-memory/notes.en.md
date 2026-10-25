# Approach

## Interviewer's Intent

"Why move to 1F1B?" "Why does a deeper pipeline with more micro-batches not blow up activation memory?" — the answer is that 1F1B compresses the number of micro-batches in flight at each stage from GPipe's $m$ down to $\min(m, p-i)$. If you can write this peak formula, you understand the **memory** half of PP (`distributed-pipeline-bubble` covers the **time** half).

## Where the In-flight Count per Segment Comes From

The 1F1B steady state is "one forward immediately followed by one backward". In the warmup phase, stages closer to the input must run more forwards while waiting to go backward; segment `i` can hold at most $\min(m,\,p-i)$ micro-batches' worth of activations:
- Input end $i=0$: $\min(m,p)$ (heaviest).
- Output end $i=p-1$: $\min(m,1)=1$ (lightest; each arrival immediately gets a backward and is released).

GPipe has no such interleaving: all $m$ forward passes finish first, and every segment accumulates $m$ copies.

## How Much Is Saved (key point)

$$ \frac{\text{offfb\_worst}}{\text{gpipe}} = \frac{\min(m,p)}{m} = \begin{cases} 1 & m\le p\\ p/m & m>p \end{cases} $$

- $m \le p$: **the heaviest segment saves nothing at all** (all m forwards are already running during warmup); only the output-end segment shrinks to 1 copy.
- $m > p$: the heaviest segment also drops to $p$ copies, a ratio of $p/m$. Hence "deepening the pipeline ($p$) actually increases memory" — $p$ is squeezed from both sides by bubbles (which want $m\gg p$) and memory (which wants small $p$), and this is the root cause of why the PP degree is hard to choose.

## Implementation Notes

$lps = L//p$ (divisibility required), $\text{per\_mb}=\text{lps}\cdot\text{act\_bytes}$, then multiply by the in-flight count of each tier. The grader's independent reference uses **per-segment profiling to take max/min** to obtain the heaviest/lightest segments (equivalent to the closed forms $\min(m,p)$/$\min(m,1)$; the two paths validate each other).

## Related Problems

`distributed-pipeline-bubble` (PP bubbles/time), `training-state-memory` (parameter/gradient/optimizer memory, which must be added to activations for total memory). Activations themselves can be further compressed with the gradient checkpointing of `transformer-mem` — an orthogonal dimension to scheduling.
