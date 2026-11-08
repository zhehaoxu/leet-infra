# Mamba2 Source Walkthrough: The SSD Chunk-Scan Three-Stage Ledger

> Upstream repository: `state-spaces/mamba` (branch `main`)
> Files involved: `mamba_ssm/ops/triton/ssd_chunk_state.py`,
> `mamba_ssm/ops/triton/ssd_state_passing.py`, `mamba_ssm/ops/triton/ssd_chunk_scan.py`,
> `mamba_ssm/ops/triton/ssd_combined.py`

## 1. Where These Rules Live in Mamba

Mamba-1 (the `la-mamba-selective-scan` problem) writes the state space scan as a
**token-by-token serial recurrence**: every step does `h = A h + B x` and then multiplies
by `C` to read out the output. The serial dependency leaves the GPU underutilized.

Mamba-2's SSD (State Space Dual) rewrites the same thing into a **chunk-parallel** form:
the `l` tokens are cut into `c = l // chunk_size` chunks, each chunk is computed in
parallel with matrix products, and only the **state** crosses chunk boundaries. Besides the
Triton kernels, upstream ships a **pure PyTorch reference implementation**, which is the
object of this problem.

The division of labour across the three stages:

| Stage | Upstream function | What it owns |
|---|---|---|
| one | `chunk_state_ref` | each position inside a chunk contributes to the state at the chunk end |
| two | `state_passing_ref` | decay-weighted propagation of the states across chunks |
| three | `chunk_scan_ref` | the output inside a chunk (cross term + initial-state term + D skip + z gate) |

```
dA = dt * A ;  dA_cumsum = cumsum(dA)          (chunk-local)
        │
        ├─ stage one: states[b][c][h][p][n]
        │            = Σ_l B · decay_states · dt · x
        │
        ├─ stage two: decay_chunk ((c+1) × (c+1) lower triangle)
        │            passed[b][c][h][p*n] / final_state[b][h][p*n]
        │
        └─ stage three: CB → scores_decay → out
```

## 2. Six Points of Stage One `chunk_state_ref`

Upstream code (`ssd_chunk_state.py:1094-1122`):

```python
def chunk_state_ref(B, x, dt, dA_cumsum):
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen <= nchunks * chunk_size
    ngroups = B.shape[2]
    assert nheads % ngroups == 0
    B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    if seqlen < nchunks * chunk_size:
        x = F.pad(x, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
        B = F.pad(B, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
    x = rearrange(x, "b (c l) h p -> b c l h p", l=chunk_size)
    B = rearrange(B, "b (c l) ... -> b c l ...", l=chunk_size)
    decay_states = torch.exp((dA_cumsum[:, :, :, -1:] - dA_cumsum))
    return torch.einsum("bclhn,bhcl,bhcl,bclhp->bchpn", B.to(x.dtype),
                         decay_states.to(x.dtype), dt.to(x.dtype), x)
```

### ★ Point 2.1: The Subtraction in `decay_states` Is "Last Minus Current"

`dA_cumsum[:, :, :, -1:] - dA_cumsum` ⇒ every entry is
`exp(dA_cumsum[lc-1] - dA_cumsum[l])`.

Measured (M=1, h=1, chunk=4): `[0.716424, 0.764340, 0.856528, 1.0]` — the
last entry is exactly `1.0`, and **the earlier the position the smaller the value**
(the decay is stronger).

Writing `dA_cumsum[l] - dA_cumsum[lc-1]` flips the direction: you get
`[1.39, 1.31, 1.17, 1.0]`, which is **greater than 1** — amplification rather than decay.

### ★ Point 2.2: The `(g h)` Flatten Is Group Major

`repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)` expands to
`head = g * (h // G) + hi`.

★ **On data with `ngroups == 1`, the `(g h)` and `(h g)` orders are exactly the same** —
this is the easiest judging trap to miss: only multi-group data can tell them apart.

### ★ Point 2.3: The einsum Sums over `l` Only

`torch.einsum("bclhn,bhcl,bhcl,bclhp->bchpn", ...)`:
the output `bchpn` contains `h` and `n`, and `h` also appears in the inputs ⇒ `h` is
**not** summed over. Only `l` is the summed index.

So `B5[b][c][l][h][n]` and `x5[b][c][l][h][p]` must use **the same** `h`; you must not
write `Σ_{l,h'}`. The two differ when there is more than one head.

### ★ Point 2.4: `dA_cumsum` Is Chunk-Local

`torch.cumsum(dA, dim=-1)` acts on the last axis of `[b][h][c][lc]` ⇒ **every chunk starts
over from zero**.

Measured: the end of chunk 0 is `-0.489734`, the start of chunk 1 is `-0.147134` —
**not** a continuation of chunk 0.

### ★ Point 2.5: The `F.pad` Branch of `seqlen` Contradicts the Downstream Assertion

`chunk_state_ref` has `if seqlen < nchunks * chunk_size: F.pad(...)`, but
`chunk_scan_ref` has `assert seqlen == nchunks * chunk_size`, and in
`ssd_chunk_scan_combined_ref` the `dt` is padded to a multiple while `seqlen` still holds
the original value.

⇒ **On that upstream reference path, `seqlen % chunk_size != 0` raises AssertionError
immediately**, and the `F.pad` branch inside `chunk_state_ref` **can never be reached**
(it only takes effect when `chunk_state_ref` is called on its own).

### ★ Point 2.6: `nheads % ngroups == 0` Is a Hard Constraint

Upstream `assert nheads % ngroups == 0`. Since `h // G` is a floor division, a
non-divisible pair would make the expanded head count differ from `nheads` and the later
indexing would go out of alignment.

## 3. Four Points of Stage Two `state_passing_ref`

Upstream code (`ssd_state_passing.py:327-350`):

```python
def state_passing_ref(states, dA_chunk_cumsum, initial_states=None):
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, 0])
    states = torch.cat([rearrange(initial_states, "b h d -> b 1 h d"), states], dim=1)
    dA_chunk_cumsum = F.pad(dA_chunk_cumsum, (1, 0))
    dA_chunk_cumsum = torch.cumsum(dA_chunk_cumsum, dim=-1)
    nchunks = dA_chunk_cumsum.shape[-1]
    dt_chunk_segment_sum = dA_chunk_cumsum[:, :, :, None] - dA_chunk_cumsum[:, :, None, :]
    decay_chunk = torch.exp(dt_chunk_segment_sum)
    causal_mask = torch.tril(torch.ones(nchunks, nchunks, device=states.device, dtype=bool), diagonal=0)
    decay_chunk = decay_chunk.masked_fill(~causal_mask, 0)
    out = torch.einsum("bhzc,bchd->bzhd", decay_chunk.to(dtype=states.dtype), states)
    return out[:, :-1], out[:, -1]
```

### ★ Point 3.1: Two Lengths Grow from `c` to `c+1`

- the `cat` puts the initial state in front ⇒ the length of `states` becomes `c+1`;
- `F.pad(dA_chunk_cumsum, (1, 0))` plus `cumsum` ⇒ the length of `dA_chunk_cumsum` becomes
  `c+1`.

So `decay_chunk` is **`(c+1) x (c+1)`**, and `nchunks = dA_chunk_cumsum.shape[-1]` takes
the **padded** length (measured 3x3 rather than 2x2 for the M=1/2-chunk case).

### ★ Point 3.2: The Matrix Must Not Be Transposed

`einsum("bhzc,bchd->bzhd")`: `decay_chunk[b][h][z][c]`, where the **row `z` is the output
chunk and the column `c` is the summed chunk**.

Measured `decay_chunk[0][0]` (c=2):

```
[1.0,      0.0,      0.0     ]
[0.61279,  1.0,      0.0     ]
[0.398209, 0.649829, 1.0     ]
```

The zeros off the diagonal of row 0 mean the initial-state column is only ever used
through the diagonal.

### ★ Point 3.3: The Split of `out[:, :-1]` and `out[:, -1]`

`passed` is the **first `c`** entries, `final_state` is the **last one**.
The length of `passed` is `c`, not `c+1`.

### ★ Point 3.4: The Default Initial State Is Zero

`torch.zeros_like(states[:, 0])` — the shape is `[b][h][d]`, not `[b][c][h][d]`.

## 4. Five Points of Stage Three `chunk_scan_ref`

Upstream code (`ssd_chunk_scan.py:1846-1887`):

```python
def chunk_scan_ref(B, C, x, dt, dA_cumsum, prev_states, D=None, z=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert B.shape == (batch, seqlen, ngroups, dstate)
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen == nchunks * chunk_size
    assert C.shape == B.shape
    B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    C = repeat(C, "b l g d -> b l (g h) d", h=nheads // ngroups)
    CB = torch.einsum("bclhn,bcshn->bchls", rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                      rearrange(B, "b (c s) h n -> b c s h n", c=nchunks))
    dt_segment_sum = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
    decay = torch.exp(dt_segment_sum)
    scores_decay = CB * rearrange(decay, "b h c l s -> b c h l s")
    causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), diagonal=0)
    scores_decay = scores_decay.masked_fill(~causal_mask, 0)
    out = torch.einsum('bchls,bhcs,bcshp->bclhp', scores_decay.to(x.dtype), dt.to(x.dtype),
                       rearrange(x, "b (c s) h p -> b c s h p", c=nchunks))
    state_decay_out = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
    out_prev = torch.einsum('bclhn,bchpn->bclhp', rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                            prev_states.to(C.dtype)) * state_decay_out
    out = out + out_prev
    out = rearrange(out, "b c l h p -> b (c l) h p")
    if D is not None:
        if D.dim() == 1:
            D = rearrange(D, "h -> h 1")
        out = out + x * D
    return out if z is None else out * F.silu(z)
```

### ★ Point 4.1: `CB` Uses the **Expanded** `C` and `B`

`B` and `C` are **overwritten** by the `repeat` at the top of the function, so the `C` used
by both `CB` and `out_prev` is the **expanded** `[b][l][h][n]`.

Measured: with `ngroups=2, nheads=4`, group 0 owns heads 0/1 and group 1 owns heads 2/3.

### ★ Point 4.2: The Direction of `dt_segment_sum`

`dA_cumsum[..., :, None] - dA_cumsum[..., None, :]` ⇒
`decay[l][s] = exp(dA_cumsum[l] - dA_cumsum[s])`.

Measured (chunk=4), positive above the diagonal and negative below:

```
[ 0.0,       0.064739,  0.178613,  0.333482  ]
[-0.064739,  0.0,       0.113874,  0.268743  ]
[-0.178613, -0.113874,  0.0,       0.154869  ]
[-0.333482, -0.268743, -0.154869,  0.0       ]
```

Because `dA < 0`, the lower triangle gives `exp <= 1` (decay) while the upper triangle
gives `exp > 1` but is masked to zero ⇒ **causal**.

### ★ Point 4.3: `state_decay_out` Is the Position-Wise Value Inside the Chunk

`rearrange(dA_cumsum, "b h c l -> b c l h 1")` ⇒ `exp(dA_cumsum[l])`,
**not** the chunk-final cumulative value. Measured for chunk=2:
`[0.855344, 0.785010]` (two different values).

### ★ Point 4.4: `D` Is an Additive Skip, and a 1-D `D` Is Reshaped to `[h][1]`

`D.dim() == 1` ⇒ `rearrange(D, "h -> h 1")` ⇒ broadcast over `headdim`.
`out = out + x * D` (**added**).

### ★ Point 4.5: `z` Is a Multiplicative `silu` Gate

`return out if z is None else out * F.silu(z)`.
`F.silu(z) = z * sigmoid(z)`, **not** `sigmoid(z)`.

## 5. Four Points of the Driver `ssd_chunk_scan_combined_ref`

Upstream code (`ssd_combined.py:687-725`):

```python
def ssd_chunk_scan_combined_ref(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, dt_softplus=False):
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]
    if seqlen % chunk_size != 0:
        dt = F.pad(dt, (0, 0, 0, chunk_size - seqlen % chunk_size))
    dt = rearrange(dt, "b (c l) h -> b h c l", l=chunk_size)
    dt = dt.float()
    if dt_bias is not None:
        dt = dt + rearrange(dt_bias, "h -> h 1 1")
    if dt_softplus:
        dt = F.softplus(dt)
    dA = dt * rearrange(A, "h -> h 1 1")
    dA_cumsum = torch.cumsum(dA, dim=-1)
    states = chunk_state_ref(B, x, dt, dA_cumsum)
    states_dtype = states.dtype
    if states.dtype not in [torch.float32, torch.float64]:
        states = states.to(torch.float32)
    states = rearrange(state_passing_ref(rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1])[0],
                       "... (p n) -> ... p n", n=dstate)
    states = states.to(states_dtype)
    out = chunk_scan_ref(B, C, x, dt, dA_cumsum, states, D=D, z=z)
    return out
```

### ★ Point 5.1: `dt_bias` Comes **Before** `softplus`

The upstream order is `dt = dt + dt_bias` and then `dt = F.softplus(dt)`.
So the final effect of `dt_bias` is `(dt + bias) * A` and then the cumsum.

★ Because there is a cumsum, the increment of `dA_cumsum` is
**`(l+1) * bias[hi] * A[hi]`** (the accumulation effect), not the constant `bias * A`.

### ★ Point 5.2: `dt_bias` Is One Scalar Per Head

`rearrange(dt_bias, "h -> h 1 1")` ⇒ broadcast to every chunk position of that head.

### ★ Point 5.3: The `(p, n)` Flatten and Restore

`rearrange(states, "... p n -> ... (p n)")` and
`rearrange(states, "... (p n) -> ... p n", n=dstate)` are a pair, and the flat index is
`p * n + ni` (`p` is the high digit, `n` the low one, C order).

### ★ Point 5.4: A Non-Multiple `seqlen` Crashes the Official Reference

`dt` is padded to a multiple, but the
`assert seqlen == nchunks * chunk_size` in `chunk_scan_ref` uses the **original
`seqlen`** ⇒ AssertionError.

Measured: `seqlen=6, chunk=4` ⇒ `AssertionError` (no message).

## 6. Three Reproduction Pitfalls in the Official Source

### Pitfall 1: `einops` Is Not Available in the Evaluation Environment

Upstream uses `einops`' `rearrange` / `repeat` everywhere. The evaluation environment has
**no einops** (measured `ModuleNotFoundError: No module named 'einops'`).

⇒ When reproducing, you must replace `rearrange` / `repeat` with the equivalent `reshape` /
`permute` / `unsqueeze`, **and check the index layout one by one**. Two traps hit this
round:

1. `repeat(B, "b l g d -> b l (g h) d", h=h//g)` is equivalent to
   `B.unsqueeze(3).expand(b,l,g,h,d).reshape(b,l,g*h,d)` (`g` first);
2. `rearrange(decay, "b h c l s -> b c h l s")` is equivalent to `permute(0,2,1,3,4)`.

### Pitfall 2: The Argument Order of `F.pad`

In `F.pad(dt, (0, 0, 0, chunk_size - seqlen % chunk_size))` the shape of `dt` is
`(b, l, h)` ⇒ the pad counts run from the **last** axis backwards: `(0,0)` acts on the `h`
axis and `(0, k)` acts on the `l` axis ⇒ **the `l` axis gets `k` zeros**.

### Pitfall 3: You Have to Count the Summed Indices of an einsum Yourself

The einsum rule is "a letter is a summed index if it appears in the inputs but not in the
output". For `"bclhn,bhcl,bhcl,bclhp->bchpn"` the only summed index is `l` — the first
attempt summed over `h` as well, which is **completely invisible on data with
`nheads == 1`** and only shows up on the multi-head cases.

## 7. The Three-Layer Verification Structure

1. **Official layer**: from the three upstream `*_ref` files, use
   `ast.get_source_segment` to extract **verbatim**, replacing only the `einops`
   `rearrange` / `repeat` with the equivalent `reshape` / `permute`.
2. **Reference layer**: `/tmp/mamba850_solution_staged.py` (pure Python nested lists plus
   `math.exp`).
3. **Oracle layer**: a pure Python implementation rewritten independently of the reference
   solution, straight from the official formulas.

Measured over **486 check items** (18 shape combinations x 4 modes x several D / z / bias
combinations):
- `official-vs-ref <= 1.435e-07` (upstream is float32, the reference is float64);
- `oracle-vs-ref = 0.000e+00` (exactly equal).

There are also **9 domain-constraint assertions** (`chunk_size <= 0`,
`nheads % ngroups != 0`, a non-multiple `seqlen`, a wrong `dt` layout, a wrong `D` shape,
`seqlen != nchunks * chunk_size`, the official crash on a non-multiple `seqlen`, and so on).

**Why float64 against float32**: upstream is float32 throughout (`dt.float()`) while the
reference uses Python `float` (float64). The difference is on the order of `1e-7`, so a
`1e-9`-level tolerance in judging is enough to separate "computed it wrong" from "precision
difference".

## 8. Three Principles for Mutation Testing

The mutation list for this round has 43 entries and ends at **43/43 CAUGHT**. The pitfalls
hit along the way:

1. **Equivalent mutants must be removed on purpose**. Two entries survived the first round
   and after checking each one they turned out to be **truly equivalent**:
   - `acc += sd * dv * xv` becoming `acc += dv * sd * x[...]` (commutativity of
     multiplication);
   - `row.append(out_intra + out_prev)` becoming `row.append(out_prev + out_intra)`
     (commutativity of addition).

   The criterion is "identical output over the entire legal input domain", not "identical
   on this set of cases".

2. **The validation layer must be split into independent defenses**. The `chunk_scan`
   check, for instance, asserts all of:
   - `out == oracle`;
   - the upper triangle of `scores_decay` is all zero;
   - `decay[l][s]` equals `exp(dA_cumsum[l] - dA_cumsum[s])`;
   - the "z turned additive" candidate **must differ**;
   - the "D turned multiplicative" candidate **must differ**.

   With only the first assertion, an implementation that writes `out * D` where
   `out + x * D` belongs might pass.

3. **Unobservable differences need dedicated cases, and the anchor must land on an input
   that really changes.** Two typical cases this round:

   - **Bidirectional assertions need carefully chosen data too**: the `(g h)` vs `(h g)`
     order assertion for the group → head expansion has two candidates that are **exactly
     equal** on data with `ngroups == 1` ⇒ the assertion falsely fires. The fix was to move
     it out of the loop and use a **deliberately multi-group** dataset
     (`nheads=4, ngroups=2`).
   - ★ **The error message is an observable difference too**: two mutants survived the
     first round — "delete the seqlen divisibility check of `chunk_state`" and "delete the
     divisibility check of `ssd_forward`". The root cause is that after deleting the check,
     the **downstream shape validation / `chunk_state` still raises**, and the domain
     case only asserted "a `ValueError` was raised" ⇒ the two could not be told apart.
     The fix has two steps:
     ① make the domain case inputs **shape-valid** (give `dt` as `(1,1,1,2)` rather than
     `(1,3,1)`), so that deleting the check is no longer masked by the shape validation;
     ② assert the **error message content** — the message of the `ssd_forward` rejection
     must carry the `ssd_forward` tag, so that it is distinguishable from the
     `seqlen must be divisible by chunk_size` that `chunk_state` reports on its behalf.

   Lesson: **"it raised an exception" is not a fine-grained enough criterion; "which stage
   raised it and what the message says" is.**

## 9. Boundaries Against the Existing Problems

- `la-mamba-selective-scan` (Mamba-1): the **token-by-token ZOH discrete recurrence** with
  a gated skip; no chunk decomposition, no `dstate`-axis state matrix, no grouped `B`/`C`
  matrices.
- `la-gla-chunked` (GLA): the **identity ledger** between the chunk-parallel form and the
  naive recurrence; the decay is a **per-key-channel scalar**, the state is `[K, V]`, and
  there is neither a `dstate` axis nor `B`/`C`.
- `la-retnet-retention-xpos-ledger` (RetNet): the **equivalence of three representations**
  (an invariant); this problem is about the **concrete three-stage computation**.
- `la-xlstm-stabilized-gate-ledger` / `la-slstm-normalizer-state-ledger` (xLSTM): gating and
  normalizer state, unrelated to the `dstate`-axis state matrix of SSD.

★ In the upstream repository `ssd_minimal.py` has been **deleted** on the `main` branch
(404); the three `*_ref` functions now live in three separate files,
`ssd_chunk_state.py` / `ssd_state_passing.py` / `ssd_chunk_scan.py`.

## 10. Index of Official Source Locations

| Symbol | Location |
|---|---|
| `chunk_state_ref` | `ssd_chunk_state.py:1094-1122` |
| the `repeat(B, ...)` of `chunk_state_ref` | `ssd_chunk_state.py:1114` |
| the `F.pad` branch of `chunk_state_ref` | `ssd_chunk_state.py:1116-1118` |
| the `decay_states` of `chunk_state_ref` | `ssd_chunk_state.py:1121` |
| the einsum of `chunk_state_ref` | `ssd_chunk_state.py:1122` |
| `state_passing_ref` | `ssd_state_passing.py:327-350` |
| the `cat` of `state_passing_ref` | `ssd_state_passing.py:339` |
| the `F.pad` plus `cumsum` of `state_passing_ref` | `ssd_state_passing.py:340-341` |
| the `tril` mask of `state_passing_ref` | `ssd_state_passing.py:347-348` |
| the einsum of `state_passing_ref` | `ssd_state_passing.py:349` |
| the return split of `state_passing_ref` | `ssd_state_passing.py:350` |
| `chunk_scan_ref` | `ssd_chunk_scan.py:1846-1887` |
| the `assert seqlen` of `chunk_scan_ref` | `ssd_chunk_scan.py:1864` |
| the `CB` of `chunk_scan_ref` | `ssd_chunk_scan.py:1868-1870` |
| the `dt_segment_sum` of `chunk_scan_ref` | `ssd_chunk_scan.py:1871-1872` |
| the `tril` mask of `chunk_scan_ref` | `ssd_chunk_scan.py:1875-1876` |
| the cross-term einsum of `chunk_scan_ref` | `ssd_chunk_scan.py:1877-1878` |
| the `state_decay_out` of `chunk_scan_ref` | `ssd_chunk_scan.py:1879` |
| the `out_prev` of `chunk_scan_ref` | `ssd_chunk_scan.py:1880-1881` |
| the `D` skip of `chunk_scan_ref` | `ssd_chunk_scan.py:1884-1886` |
| the `z` gate of `chunk_scan_ref` | `ssd_chunk_scan.py:1887` |
| `ssd_chunk_scan_combined_ref` | `ssd_combined.py:687-725` |
| the `F.pad` of `ssd_chunk_scan_combined_ref` | `ssd_combined.py:703-704` |
| the `dt_bias` of `ssd_chunk_scan_combined_ref` | `ssd_combined.py:707-708` |
| the `softplus` of `ssd_chunk_scan_combined_ref` | `ssd_combined.py:709-710` |
| the `dA` of `ssd_chunk_scan_combined_ref` | `ssd_combined.py:711-712` |
