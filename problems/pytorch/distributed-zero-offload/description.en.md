# ZeRO-Offload: The Tradeoff Between GPU Memory and PCIe Bandwidth

## Problem

`training-state-memory` computes how much memory ZeRO partitioning saves, and `distributed-zero-communication` computes how much **cross-GPU
bandwidth** it costs. ZeRO-Offload takes another route: it **offloads optimizer states (and gradients) to CPU memory**, trading **PCIe bandwidth for GPU memory**—
so even a single GPU can train large models. This problem accounts for that exchange (pure computation, custom simplified model).

## Model

Let the parameter count be $N$. Bytes per parameter in the training state (aligned with `training-state-memory`):

| Component | Bytes per parameter | Total bytes |
|---|---|---|
| fp16 parameters | `param_bytes` (default 2) | $2N$ |
| fp16 gradients | `grad_bytes` (default 2) | $2N$ |
| fp32 optimizer states (master+momentum+variance) | `opt_bytes` (default 12) | $12N$ |

ZeRO-Offload's placement policy: **only the fp16 parameters reside on the GPU**; gradients are sent to the CPU as soon as they are computed, and optimizer states live entirely on the CPU. Thus:

- **GPU memory** (training-state part) $= \texttt{param\_bytes}\cdot N$
- **CPU memory** (offloaded part) $= (\texttt{grad\_bytes}+\texttt{opt\_bytes})\cdot N$
- **PCIe traffic per iteration**: gradients GPU→CPU ($\texttt{grad\_bytes}\cdot N$) $+$ updated fp16 parameters CPU→GPU
  ($\texttt{param\_bytes}\cdot N$) $= (\texttt{grad\_bytes}+\texttt{param\_bytes})\cdot N$

The PCIe transfer time is $t_{\text{pcie}} = \dfrac{\text{pcie\_bytes}}{\texttt{pcie\_bandwidth}}$. If it is **greater than** the GPU
per-iteration compute time `compute_time`, PCIe becomes the bottleneck (computation can no longer hide the transfer): $\text{bottleneck} = (t_{\text{pcie}} > \texttt{compute\_time})$.

## Rules

Implement `solve(n_params, pcie_bandwidth, compute_time, param_bytes=2, grad_bytes=2, opt_bytes=12)`, returning
`(gpu_mem, cpu_mem, pcie_bytes, is_bottleneck)`: the first three are float byte counts, the last a bool.

## Examples

```
N=1e9, pcie_bandwidth=3.2e10 (32 GB/s), compute_time=0.1s, default bytes
gpu_mem   = 2·1e9  = 2.0e9
cpu_mem   = 14·1e9 = 14.0e9
pcie_bytes= 4·1e9  = 4.0e9
t_pcie    = 4.0e9 / 3.2e10 = 0.125s  >  0.1  → is_bottleneck = True
→ (2.0e9, 14.0e9, 4.0e9, True)
```

## Constraints

- `n_params ≥ 1`; `pcie_bandwidth > 0`; `compute_time ≥ 0`
- Return `(float, float, float, bool)`; byte counts compared with relative error $10^{-6}$
- Ignore activation memory, bidirectional CPU↔GPU concurrency, and communication/computation overlap optimizations—compute purely according to the definition above
- `is_bottleneck` uses **strictly greater than** ($t_{\text{pcie}} == \texttt{compute\_time}$ counts as `False`)
