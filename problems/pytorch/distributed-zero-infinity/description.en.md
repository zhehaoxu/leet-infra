# ZeRO-Infinity: HBM / DRAM / NVMe Three-Tier Offload Accounting

## Problem

ZeRO-Offload moves optimizer states to **CPU memory** (two tiers). ZeRO-Infinity adds another tier, **NVMe SSD**, and shards the offloaded data
**along the data-parallel dimension across ranks** (each rank stores only $1/d$ of the offloaded data instead of every node holding the full copy), thereby training very large models with nearly unlimited storage. This problem accounts for how much each of the three tiers holds, and how many bytes must be moved from lower tiers into GPU memory in one step (pure computation, custom simplified model).

## Model (follow this definition exactly)

Parameter count $N$, data-parallel degree $d$. The training state has three parts, **all sharded along $d$ as in ZeRO-3**, with each rank holding $1/d$:

$$ P = \frac{\texttt{param\_bytes}\cdot N}{d},\quad G = \frac{\texttt{grad\_bytes}\cdot N}{d},\quad O = \frac{\texttt{opt\_bytes}\cdot N}{d} $$

Placement policy (two switches + one target tier):

- **The gradient shard $G$ always resides in HBM** (used immediately by the backward computation each step, never offloaded).
- `offload_param=True` → the parameter shard $P$ leaves HBM and lands in the `offload_target` tier; otherwise it resides in HBM.
- `offload_optimizer=True` → the optimizer shard $O$ leaves HBM and lands in the `offload_target` tier; otherwise it resides in HBM.
- `offload_target ∈ {"dram","ssd"}`: all offloaded shards go to the same lower tier.

Thus, per rank:

- `hbm_bytes` $= G + \mathbb{1}[\neg\text{offload\_param}]\cdot P + \mathbb{1}[\neg\text{offload\_optimizer}]\cdot O$
- Offloaded bytes $= \mathbb{1}[\text{offload\_param}]\cdot P + \mathbb{1}[\text{offload\_optimizer}]\cdot O$; they all go into `dram_bytes` (target=dram) or `ssd_bytes` (target=ssd), and the other tier is 0
- `step_transfer_bytes` $=$ offloaded bytes (each step these shards are prefetched from the lower tier into HBM; this model does not add gradient write-back and ignores overlap/caching)

## Rules

Implement `solve(n_params, dp_degree, offload_param, offload_optimizer, offload_target, param_bytes=2, grad_bytes=2, opt_bytes=12)`, returning the dict `{hbm_bytes, dram_bytes, ssd_bytes, step_transfer_bytes}` (all float byte counts).

## Examples

```
N=7e9, d=8, offload_param=True, offload_optimizer=True, offload_target="ssd", default bytes
P = 2·7e9/8 = 1.75e9,  G = 1.75e9,  O = 12·7e9/8 = 10.5e9
hbm_bytes          = G = 1.75e9
dram_bytes         = 0
ssd_bytes          = P + O = 1.225e10
step_transfer_bytes= P + O = 1.225e10
→ {"hbm_bytes":1.75e9, "dram_bytes":0.0, "ssd_bytes":1.225e10, "step_transfer_bytes":1.225e10}
```

## Constraints

- `n_params ≥ 1`; `dp_degree ≥ 1`; `offload_target ∈ {"dram","ssd"}`; default byte widths $2/2/12$
- Return four float byte counts, compared with relative error $10^{-6}$
- Ignore activation memory, PCIe/NVMe bandwidth and computation overlap, and offload granularity scheduling—compute purely according to the definition above
- With both `offload_param` and `offload_optimizer` False, this is ZeRO-3 with everything in HBM: `hbm_bytes = (P+G+O)`, `step_transfer_bytes = 0`
