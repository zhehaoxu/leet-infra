# FunSearch Island Programs Database: Cluster Registration, Temperature Sampling and Weak-Island Reset

FunSearch keeps prompting an LLM to rewrite a single small function, but what actually decides where
the search goes is not the model — it is the **programs database** behind it. Programs are clustered
by their per-test score vector, sampling probabilities come from a softmax over `cluster score / temperature`,
and the temperature decays linearly with the number of programs that island has seen; a pool of islands
runs in parallel, and the weaker half is wiped periodically and seeded with a clone of a stronger
island's best program. This problem turns `google-deepmind/funsearch`'s `ProgramsDatabase` into a pure
function ledger.

## Interface

```python
def solve(events, num_islands=10, functions_per_prompt=2,
          cluster_sampling_temperature_init=0.1,
          cluster_sampling_temperature_period=30000,
          function_name="f") -> dict
```

`events` is a sequence of three kinds:

| `kind` | fields | meaning |
| --- | --- | --- |
| `register` | `program`, `scores`, `island` | register one program into an island; `island=None` marks a seed program broadcast to every island |
| `prompt` | `island`, `clusters`, `programs` | take one prompt from that island; `clusters`/`programs` are the **explicit sampling outcome** |
| `reset` | `founders` | reset the weaker half of islands; `founders` gives the founder island per reset island |

`program` is `{"id": str, "doc": str, "size": int}` — `size` is the source length, used as the weight
inside a cluster.

The result carries `clusters` (per island, each with `signature`/`score`/`programs`), `cluster_count`,
`num_programs`, `best_score`, `best_program`, `registrations`, `prompts` and `resets`. Each
`prompts[i]` records `temperature`, `cluster_scores`, `probabilities`, `chosen_clusters`,
`chosen_programs`, `program_probs`, `sorted_clusters`, `versions`, `version_generated` and `header`.

## Input and Output

```
Input:  num_islands=2, functions_per_prompt=2, function_name="f"
        events=[{kind:"register", island:0,
                 program:{id:"p0", doc:"returns 0", size:12},
                 scores:{"t2": 9.0, "t1": 1.0}},
                {kind:"prompt", island:0, clusters:[0], programs:[0]}]
Output: clusters[0] = [{signature:[1.0], score:1.0, programs:["p0"]}]
        best_score[0] = 1.0
        prompts[0].header = {name:"f_v1", docstring:"Improved version of `f_v0`."}
        prompts[0].version_generated = 2
```

The cluster key of `{"t2": 9.0, "t1": 1.0}` is the **test-name-sorted** `[1.0]`, while the cluster
score is the value of the test inserted **last** (`t1`, i.e. `1.0`) — not the visually larger `9.0`.
That single line is the easiest thing to get wrong here.

## Rules

- **Cluster key**: `tuple(scores[k] for k in sorted(scores))` — per-test scores in ascending test-name
  order. Two programs share a cluster only when **every** per-test score matches; a different score
  vector is a different cluster.
- **Cluster score**: `scores[list(scores)[-1]]`, the score of the test inserted **last**. It is fixed
  when the first member enters the cluster; later members only append to `programs` and **never**
  change it, even when their own reduced score is higher.
- **Broadcast**: `island=None` registers into every island once, bumping each island's `num_programs`.
- **Island best**: only `score > current best` replaces the incumbent (strictly greater; ties keep the
  earlier program), and it also stores the winning per-test scores, which the reset path needs.
- **Temperature**: `T = t_init * (1 - (num_programs % period) / period)`, where `num_programs` counts
  registrations on **that island** (broadcasts and reset clones included). It returns to `t_init`
  whenever `n % period == 0`.
- **Cluster probabilities**: `softmax(cluster_scores / T)`, in cluster insertion order.
- **How many clusters**: `min(len(clusters), functions_per_prompt)`, sampled **with replacement**
  (the same cluster may appear twice).
- **Version numbering**: the sampled clusters are ordered by **ascending cluster score** and named
  `f_v0 … f_v(n-1)`; `f_v0` keeps its original docstring and `f_vi` (i≥1) gets
  `Improved version of \`f_v{i-1}\`.`. A final empty header `f_vn` is appended with docstring
  `Improved version of \`f_v{n-1}\`.`, and `version_generated = n + 1`.
- **In-cluster sampling**: `normalized = (size - min) / (max + 1e-6)` then `softmax(-normalized, 1.0)`.
  The denominator is `max + 1e-6`, not `max - min` — copied verbatim from the official implementation,
  so shorter programs are favoured without collapsing to `argmin`.
- **Reset**: sort islands by `(island best, island id)` ascending and take the first
  `num_islands // 2` as the weak half (`num_islands=1` resets nothing). Each weak island is emptied
  and its best set to `-inf`, then the founder's `best_program` is cloned in, so the new island holds
  exactly one cluster with `num_programs=1` and inherits the founder's score.
- **Validation**: `num_islands`/`functions_per_prompt`/`period` must be integers `>=1` (`bool`
  rejected); `t_init` must be finite and positive; scores must be finite and not `bool`; `scores`
  must be non-empty. A prompt from an empty island, an out-of-range cluster or program index, a
  `clusters` length other than `min(len(clusters), fpp)`, a founder outside the kept half or without
  a program, a `founders` length other than `num_islands // 2`, or an unknown `kind` all raise
  `ValueError`.

## Hints

- The same `scores` mapping in a different insertion order yields a **different cluster score**
  (key order is unchanged, the score is not). Use `list(scores)[-1]`; do not reach for `max` or
  `sorted`.
- Temperature belongs to the **island**, not the database: a broadcast seed bumps every island's
  counter, and a reset restarts the new island at 0 — both shift later sampling probabilities.
- `-0.0` and `0.0` are the **same dict key** in Python (equal `hash` and `==`), so `{"t": -0.0}` and
  `{"t": 0.0}` land in one cluster. That is the intended alignment, not a bug.
