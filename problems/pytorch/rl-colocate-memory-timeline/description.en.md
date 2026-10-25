# RL Systems: Colocated Multi-Model Memory and Phase Timeline Accounting

## Problem

One RLHF step houses several models at once: the actor needs training, the critic needs training, the reference and reward models only do inference. Putting them on the **same GPU** (colocation) forces movement between phases: while generating, offload the ref/critic weights to the CPU so that KV cache and activations fit; when `ref_logp` comes around, swap ref back in. This is exactly what verl / OpenRLHF's `offload` switches manage, and both OOMs and step latency are decided by this ledger.

Here you are handed an **already written schedule table** — per-model byte counts, which phase uses which model on which device and whether it trains, plus the residency flags per phase per model — and you must produce the ledger. No real forward pass anywhere.

## Model and conventions

$M$ models, $P$ phases, $D$ devices. Conventions:

- `resident[p][m] == 1` means the model's **full training state** ($\text{param}_m+\text{opt}_m$) currently sits on `model_device[m]`; $0$ means the whole block was recycled to the CPU side (offloaded). Optimizer state moves **together with** the parameters, which is the block-offload convention used by the frameworks; inference-only models carry `opt = 0`.
- Activations appear transiently only where an update happens: if `phase_train[p]==1`, then `phase_device[p]` additionally carries $\text{act}_{\text{phase\_model}[p]}$. Inference phases run no-grad in chunks, so their activations are not counted (KV cache belongs to `kv-cache-memory`).

Occupancy of phase $p$ on device $d$:

$$\text{occ}[p][d]=\!\!\sum_{m:\ \text{model\_device}[m]=d,\ \text{resident}[p][m]=1}\!\!(\text{param}_m+\text{opt}_m)\ +\ \mathbb{1}[\text{phase\_train}[p]=1\wedge \text{phase\_device}[p]=d]\cdot \text{act}_{\text{phase\_model}[p]}$$

## Rules

Nine inputs: `models` $(M,3)$ int64 (columns `param_bytes, opt_bytes, act_bytes`), `model_device` $(M,)$, `phase_model` $(P,)$, `phase_device` $(P,)$, `phase_train` $(P,)\in\{0,1\}$, `resident` $(P,M)$ of $0/1$, `phase_time` $(P,)$, the scalar `hbm_cap` and the scalar `num_devices` $=D$.

1. **Peak**: `peak[d]` $=\max_p$ `occ[p][d]` ($0$ when there are no phases).
2. **Transfer volume**: the timeline starts from the phase-0 configuration as the **steady state**, and a step must end back in that same steady state (otherwise the generation phase of the next step starts from the wrong configuration). Hence the flip count of model $m$ is
   $$\text{moves}_m=\#\{p\in[1,P): \text{resident}[p][m]\ne\text{resident}[p-1][m]\}+\mathbb{1}[\text{resident}[P-1][m]\ne\text{resident}[0][m]]$$
   and `offload[model_device[m]]` $+=$ $\text{moves}_m\cdot(\text{param}_m+\text{opt}_m)$. Mind that closing term: if the last phase differs from the steady state it has to be **moved once more**.
3. **Idle windows**: `idle_time[d]` $=\sum_p \text{phase\_time}[p]\cdot\mathbb{1}[\text{phase\_device}[p]\ne d]$; `max_idle[d]` $=$ the duration sum of that device's **longest contiguous** run of idle phases (within one step, no wrap-around).
4. **Feasibility**: `feasible[d]` $=$ (`peak[d] <= hbm_cap`).

Return the 5-tuple: `peak` int64 $(D,)$, `offload` int64 $(D,)$, `idle_time` int64 $(D,)$, `max_idle` int64 $(D,)$, `feasible` bool $(D,)$.

## Example

4 models (actor/critic/ref/reward), 6 phases, 2 devices:

```
models(row order = actor,critic,ref,reward) = [[400,2000,300],[400,2000,250],[400,0,120],[400,0,110]]
model_device = [0, 1, 0, 1]                      # actor+ref on GPU0, critic+reward on GPU1
phases: p0 generate(actor,GPU0,infer,t=5)  p1 score(reward,GPU1,infer,t=2)
        p2 value(critic,GPU1,infer,t=2)    p3 old_logp(actor,GPU0,infer,t=2)
        p4 ref_logp(ref,GPU0,infer,t=2)    p5 actor-update(actor,GPU0,train,t=4)
phase_model=[0,3,1,0,2,0]  phase_device=[0,1,1,0,0,0]  phase_train=[0,0,0,0,0,1]
phase_time =[5,2,2,2,2,4]  hbm_cap=4000  num_devices=2

resident = [[1,0,0,1],      # generate: actor resident, critic/ref parked on CPU, reward on GPU1
            [1,0,0,1],      # score
            [1,1,0,1],      # value: swap critic in
            [1,1,0,1],      # old_logp
            [1,1,1,0],      # ref_logp: swap ref in, offload reward
            [1,1,1,0]]      # actor update
```

Occupancy matrix (each row = `[GPU0, GPU1]`):

```
p0 [2400,  400]   p1 [2400,  400]   p2 [2400, 2800]
p3 [2400,  2800]  p4 [2800, 2400]   p5 [3100, 2400]   # p5 = 2400+400+act300
→ peak      = [3100, 2800]
→ offload   = [800, 5600]     # critic 2 flips × 2400, ref 2 × 400, reward 2 × 400, actor 0
→ idle_time = [4, 13]         # total 17; GPU0 busy 5+2+2+4=13, GPU1 busy 2+2=4
→ max_idle  = [4, 8]          # GPU1's idle splits into runs of 5 and 2+2+4=8
→ feasible  = [True, True]    # with hbm_cap=3000 it becomes [False, True]
```

## Constraints

- CPU; $1\le M\le 16$, $0\le P\le 64$, $1\le D\le 8$; all byte counts and times are non-negative int64; `resident` only holds $0/1$; indices in `model_device`, `phase_model`, `phase_device` must be in range.
- Checking: the first four outputs with `torch.equal`, `feasible` with `torch.equal` on a bool tensor.
- Degenerate invariants: `resident` all $1$ $\Rightarrow$ `offload` all $0$; $D=1$ $\Rightarrow$ `idle_time` and `max_idle` all $0$; `phase_train` all $0$ $\Rightarrow$ no `act` inside any `peak`; $P=0$ $\Rightarrow$ the four integer outputs are $0$ and `feasible` all `True`; `hbm_cap=0` with any resident model $\Rightarrow$ `feasible` all `False`.
- Division of labour: `training-state-memory` computes the 16-bytes-per-parameter training state of a **single model** and its ZeRO sharding; `distributed-zero-offload` trades PCIe bandwidth against memory for a **single** offloaded model; this problem is the **multi-model × multi-phase** colocation timeline (who holds how much when, how many bytes move, who sits idle). Advantage/loss formulas live in `to-*`, token-load re-sharding in `rl-rollout-shard-repack`.
