# Preemption Decision: Swap-Out vs Recompute

## Problem

The KV cache pool of an online service has a fixed size. When a new request's prefill needs room and no slot is free, the engine has to pick victims among the running requests. Having picked one, there is a second choice:

- **swap**: move that request's KV to host memory, freeing device slots, and copy it back later to resume decoding. The cost is two transfers: `2 × kv_tokens × bytes_per_token / bandwidth`.
- **recompute**: drop the KV and re-prefill the whole request later (prompt plus every token already generated). The cost is `(prompt_len + generated) × ticks_per_token`.

Short sequences with little generation are cheaper to recompute; long or heavily generated ones are cheaper to swap. Given the current ledger of running requests, output whom to preempt, which path each takes, how many slots were freed, and the total cost in ticks.

## Rules

1. **Who can be preempted**: only requests with `freezable[i] == True` are candidates. The rest are protected (say, finishing or covered by an SLA) and are never touched, no matter how much they would free.
2. **Priority tiers**: candidates are processed tier by tier, from the lowest `priority` upward. **Within a tier**, pick the larger `kv` first (frees more per victim); equal `kv` breaks on the smaller index. A higher tier is only considered after the current tier is exhausted.
3. **Stop when enough**: as soon as the cumulative freed slots reach `need`, stop — no "while we are at it" extras.
4. **Per-victim comparison**:
   - `swap_ticks = ceil(2 * kv[i] * bytes_per_token / swap_bw)`
   - `recompute_ticks = (prompt[i] + generated[i]) * tick_per_token`
   - choose `"swap"` when `swap_ticks <= recompute_ticks`, otherwise `"recompute"`.
5. **Infeasible**: if preempting every candidate still cannot reach `need`, give up entirely — no preemption happens, `ticks` is `-1`, `replay_tokens` is 0, `victims` and `modes` are empty, and `freed` still reports how much could have been freed.

Return a five-tuple:

- `victims`: indices of preempted requests, in the order they were chosen;
- `modes`: `"swap"` / `"recompute"`, aligned with `victims`;
- `freed`: slots actually freed (may exceed `need`);
- `ticks`: total cost of all preempted requests; `-1` when infeasible;
- `replay_tokens`: tokens that must be re-prefilled by the recompute victims; 0 when infeasible.

## Examples

**Example 1** (both modes inside one tier): `kv = [4, 40, 40]`, `prompt = [30, 5, 5]`, `generated = [10, 100, 1]`, `priority = [0, 0, 0]`, `need = 44`, `bytes_per_token = 4`, `swap_bw = 16`, `tick_per_token = 1`, all freezable

```
candidates by kv descending: 1(40), 2(40), 0(4)
request 1: swap = ceil(2*40*4/16) = 20, recompute = (5+100)*1 = 105  -> swap, 20 ticks
           freed 40 < 44, continue
request 2: swap = 20, recompute = (5+1)*1 = 6                        -> recompute, 6 ticks, 6 replay tokens
           freed 80 >= 44, stop (request 0 is untouched)
```

```
victims       = [1, 2]
modes         = ["swap", "recompute"]
freed         = 80
ticks         = 26
replay_tokens = 6
```

**Example 2** (move up a tier only when needed): `kv = [30, 6, 50]`, `prompt = [10, 10, 10]`, `generated = [2, 2, 2]`, `priority = [2, 0, 1]`, `need = 20`, `bytes_per_token = 4`, `swap_bw = 64`, `tick_per_token = 1`

```
priority 0: only request 1 (kv=6) -> swap, ceil(48/64)=1 tick, freed 6 < 20
priority 1: request 2 (kv=50)     -> swap, ceil(400/64)=7 ticks, freed 56 >= 20, stop
```

```
victims = [1, 2]
modes   = ["swap", "swap"]
freed   = 56
ticks   = 8
replay_tokens = 0
```

**Example 3** (protected requests stay out): `kv = [30, 30, 10]`, `prompt = [8, 8, 4]`, `generated = [2, 2, 1]`, `priority = [0, 1, 2]`, `need = 20`, `bytes_per_token = 4`, `swap_bw = 64`, `tick_per_token = 1`, `freezable = [True, False, False]`

```
only request 0 is preemptible: swap = ceil(2*30*4/64) = 4 ticks, frees 30 >= 20
```

```
victims = [0]
modes   = ["swap"]
freed   = 30
ticks   = 4
replay_tokens = 0
```

## Constraints

- `n = len(kv) = len(prompt) = len(generated) = len(priority) = len(freezable)`, `1 <= n <= 64`.
- `kv[i] >= 1`, `prompt[i] >= 1`, `generated[i] >= 0`, `need >= 1`, `bytes_per_token >= 1`, `swap_bw >= 1`, `tick_per_token >= 1`.
- Everything is integer arithmetic: use an integer ceiling, never a float division followed by rounding.
- Only the numbers and list contents are judged; any implementation style is fine.
