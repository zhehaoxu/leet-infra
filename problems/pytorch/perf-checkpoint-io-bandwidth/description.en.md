# Checkpoint I/O Bandwidth and Asynchronous Persistence

## Problem

Training crashes, and the more frequent the **checkpoints**, the less work is lost on restart - but every save
carries the whole training state from HBM to host memory and on to the parallel filesystem / NVMe, so a **slow
store stalls the loop step by step**. **Asynchronous saving** (Megatron's `async_save`) splits "copy into pinned
staging" from "write to storage": only the former blocks training, while the write overlaps later steps, buying
lag space with B staging buffers. How often to save is a classic trade-off with a closed form (Young-Daly /
CheckFreq): interval $\approx\sqrt{2CF}$, where $C$ is the time one checkpoint **blocks** the loop and $F$ is the
mean time between failures - asynchrony shrinks $C$ from "copy + write" to "copy only", which in turn allows a
shorter interval.

`training-state-memory` says how many bytes must move, `distributed-zero-offload` accounts for the read side
(per-step HBM↔DRAM prefetch during training) and `distributed-sublinear-checkpoint` for the recompute side;
this problem is only about the **write-side I/O bill**.

## Rules

Let $N=$ `n_params`. The training-state byte count follows the mixed-precision + Adam convention of
`training-state-memory` (fp16 param 2 + fp16 grad 2 + fp32 master 4 + $m$ 4 + $v$ 4):

$$ \text{state\_bytes}=N\cdot\texttt{state\_bytes\_per\_param}\quad(\texttt{state\_bytes\_per\_param}=16\text{ by default}) $$

Divide by each of the two bandwidths once (millisecond scale):

$$ \text{copy\_ms}=\frac{1000\cdot\text{state\_bytes}}{\texttt{device\_to\_host\_bw}},\qquad
   \text{write\_ms}=\frac{1000\cdot\text{state\_bytes}}{\texttt{storage\_bw}},\qquad
   \text{period\_ms}=\texttt{checkpoint\_interval}\cdot\texttt{step\_ms} $$

- **Blocking time**: with `async_save=False`, $\text{stall}=\text{copy}+\text{write}$; with `async_save=True`
  the write overlaps training and only the pinned copy blocks, so $\text{stall}=\text{copy}$.
- **Checkpoint overhead**: $\text{overhead}=\text{stall}/\text{period}$ (it may exceed 1, meaning saving is slower
  than training).
- **Writer-can't-keep-up test**: a staging slot is held from the start of the copy until the write finishes, so B
  buffers tolerate one checkpoint lagging by at most B periods:
  $$ \text{writer\_limited}=\big(\text{copy}+\text{write}>\texttt{staging\_buffers}\cdot\text{period}\big) $$
  (strictly greater; equality is not "limited". The same test applies in both sync and async mode - in sync mode it
  is only an advisory indicator.)
- **Optimal interval** (Young-Daly / CheckFreq): with the blocking cost measured in steps
  $C=\text{stall}/\texttt{step\_ms}$ and $F=$ `mtbf_steps`:
  $$ \text{optimal\_interval\_steps}=\max\!\Big(1,\ \Big\lfloor\sqrt{2CF}\ \Big\rfloor\Big) $$

## Examples

```
Input: n_params=1e9, step_ms=100, checkpoint_interval=200, device_to_host_bw=64e9, storage_bw=10e9,
       async_save=True, staging_buffers=2, mtbf_steps=1e6, state_bytes_per_param=16
  state_bytes = 1e9 * 16 = 16,000,000,000 (~14.9 GiB)
  copy_ms = 1000*16e9/64e9 = 250.0   write_ms = 1000*16e9/10e9 = 1600.0   period_ms = 200*100 = 20000.0
  stall = 250.0 (async pays only the copy)  overhead = 250/20000 = 0.0125 (1.25%)
  writer_limited: 250+1600=1850 > 2*20000=40000? no
  optimal = floor(sqrt(2*(250/100)*1e6)) = floor(sqrt(5e6)) = 2236
Output: (16000000000, 250.0, 1600.0, 250.0, 20000.0, 0.0125, False, 2236)

Input: same as above but async_save=False
  stall = 250 + 1600 = 1850.0   overhead = 1850/20000 = 0.0925 (9.25%)
  optimal = floor(sqrt(2*18.5*1e6)) = floor(sqrt(3.7e7)) = 6082
Output: (16000000000, 250.0, 1600.0, 1850.0, 20000.0, 0.0925, False, 6082)
```

## Constraints

- `n_params` $\ge 1$; `step_ms` $>0$; `checkpoint_interval` $\ge 1$; both bandwidths $>0$;
  `staging_buffers` $\ge 1$; `mtbf_steps` $>0$; `state_bytes_per_param` $\ge 1$ (default 16)
- Return the 8-tuple `(state_bytes, copy_ms, write_ms, stall_ms, period_ms, overhead_fraction,
  writer_limited, optimal_interval_steps)`: items 1 and 8 are ints, item 7 a bool, items 2-6 floats compared with
  `atol=1e-5, rtol=1e-5`
- Single machine, single storage path: only the simplified "blocking copy + overlapped write + B-slot lag" model is
  used; many ranks contending for one filesystem are not simulated
