# Mamba2 SSD Chunk-Scan Three-Stage Ledger

## Background

Mamba-2 (`state-spaces/mamba`, branch `main`) rewrites the "selective state space scan" of
Mamba-1 from a **token-by-token serial recurrence** into a **chunk-parallel** three-stage
pipeline. Besides the Triton kernels, the upstream repo ships a **pure PyTorch reference
implementation** (`ssd_chunk_state.py::chunk_state_ref`,
`ssd_state_passing.py::state_passing_ref`, `ssd_chunk_scan.py::chunk_scan_ref`) plus the
driver that wires the three stages together
(`ssd_combined.py::ssd_chunk_scan_combined_ref`).

The pipeline consists of three stages, each responsible for one thing:

1. **Stage one `chunk_state_ref`**: computes, for every position inside a chunk, its
   contribution to the state at the end of that chunk, producing a state tensor of shape
   `[b, nchunks, nheads, headdim, dstate]`.
2. **Stage two `state_passing_ref`**: propagates the per-chunk states forward with
   **decay weighting**, producing the "incoming state" of each chunk plus the final state.
3. **Stage three `chunk_scan_ref`**: computes the output inside each chunk =
   "intra-chunk pairwise cross term" + "initial-state term", then adds the `D` skip
   connection and the `z` gate.

This task asks you to rebuild those three ledgers in **pure Python**, and to handle the
several details of the official implementation that **you will get wrong unless you read
the source**.

## Conventions

- All inputs are **nested lists** (not flattened one-dimensional lists), and the shapes must
  match strictly.
- Everything is computed with Python `float` (float64).
- **Symbols and dimensions** (copied from upstream):

  | Symbol | Shape | Meaning |
  |---|---|---|
  | `b` | int | batch |
  | `l` | int | seqlen |
  | `h` | int | nheads |
  | `p` | int | headdim |
  | `g` | int | ngroups (`h % g == 0` must hold) |
  | `n` | int | dstate |
  | `c` | int | nchunks = `l // chunk_size` |
  | `lc` | int | chunk_size |

- **`x`**: `[b][l][h][p]`.
- **`dt`** (timestep size): `[b][l][h]` (raw layout) or `[b][h][c][lc]` (the official chunk
  layout). `chunk_state` / `chunk_scan` require the **official chunk layout**
  `[b][h][c][lc]`; `ssd_forward` accepts the **raw layout** `[b][l][h]` and performs the
  chunk split itself.
- **`A`**: `[h]`, one scalar decay coefficient per head (upstream convention: negative).
- **`B_mat` / `C_mat`**: `[b][l][g][n]`.
- **`D`**: `[h]` or `[h][p]`, or `None`.
- **`z`**: `[b][l][h][p]`, or `None`.
- **`dt_bias`**: `[h]`, or `None`.
- **`chunk_size`**: a positive integer.
- **Every illegal input raises `ValueError`** (the message must name the violated
  constraint).

## mode 1: `chunk_state` —— The Intra-Chunk State Ledger

```python
solve("chunk_state", x=..., dt=..., A=..., B_mat=..., chunk_size=...)
```

Outputs:

```json
{
  "dA":          [[[[...]]]],
  "dA_cumsum":   [[[[...]]]],
  "decay_states":[[[[...]]]],
  "expanded_B":  [[[[...]]]],
  "states":      [[[[[...]]]]],
  "nchunks": 2, "chunk_size": 4, "dstate": 4, "headdim": 3
}
```

- `dA[b][h][c][l]` = `dt[b][h][c][l] * A[h]`.
- `dA_cumsum[b][h][c][l]` = the prefix sum of `dA` inside that chunk (**chunk-local, it does
  not carry across chunks**).
- `decay_states[b][h][c][l]` = `exp(dA_cumsum[b][h][c][lc-1] - dA_cumsum[b][h][c][l])`.
- `expanded_B[b][l][head][n]`: the group axis of `B_mat` expanded onto the head axis.
- `states[b][c][h][p][n]`.

### ★ Point 1.1: `decay_states` Is the Decay Measured **from the End of the Chunk**

Upstream writes `decay_states = torch.exp((dA_cumsum[:, :, :, -1:] - dA_cumsum))`,
so the subtraction is **last minus current**, hence:

- the last position's `decay_states` is **exactly `1.0`** (`exp(0)`);
- the earlier the position, the stronger the decay (because `dA < 0` makes `dA_cumsum`
  monotonically decreasing, so the last element is the smallest).

**This is the opposite convention to "accumulated from the start of the chunk"**, and it is
very easy to get backwards.

### ★ Point 1.2: The group → head Expansion Order Is `(g h)`, i.e. Group Major

Upstream writes `repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)`.
After flattening `(g h)`, **`g` is the high digit and `h` the low one**, so:

```
head index = g * (h // G) + hi        (hi ∈ [0, h//G))
```

that is, group `g` owns the head range `[g*(h//G), (g+1)*(h//G))`.
Writing `(h g)` (head major) gives **exactly the same result on data with
`ngroups == 1`**, so only multi-group data can tell the two apart.

### ★ Point 1.3: The einsum of `states` Sums over the Chunk Position Only

Upstream writes:

```python
torch.einsum("bclhn,bhcl,bhcl,bclhp->bchpn", B, decay_states, dt, x)
```

The output `bchpn` contains `h` and `n`, and `h` also appears in the inputs ⇒ **`h` is not
summed over**, only `l` is. In other words `B` and `x` must use **the same output head**;
you must not sum once more over the heads (which changes the result when there is more than
one head).

### ★ Point 1.4: `dA_cumsum` Is Chunk-Local, It Does Not Carry Across Chunks

`torch.cumsum(dA, dim=-1)` acts on the last axis of `[b][h][c][lc]`, so **every chunk starts
over from zero**. The first element of chunk `c+1` is **not** the continuation of the last
element of chunk `c`.

## mode 2: `state_passing` —— The Inter-Chunk State Passing Ledger

```python
solve("state_passing", states=..., dA_chunk_cumsum=..., initial_states=None)
```

Outputs:

```json
{
  "decay_chunk":  [[[[...]]]],
  "passed":       [[[[...]]]],
  "final_state":  [[[...]]],
  "nchunks": 2
}
```

- `states`: `[b][c][h][d]`, where `d = p * n` (the `(headdim, dstate)` pair already
  flattened).
- `dA_chunk_cumsum`: `[b][h][c]`, the last `dA_cumsum` of each chunk.
- `initial_states`: `[b][h][d]` or `None` (all zeros when `None`).
- `decay_chunk`: `[b][h][c+1][c+1]`.
- `passed`: `[b][c][h][d]` (the "incoming state" of each chunk).
- `final_state`: `[b][h][d]`.

### ★ Point 2.1: `decay_chunk` Has Size `(c+1)`, Not `c`

Upstream first concatenates the initial state in front of the states (the length becomes
`c+1` after the `cat`), then pads `dA_chunk_cumsum` with `F.pad(..., (1, 0))` and takes the
`cumsum` (its length also becomes `c+1`), and then builds a `(c+1) x (c+1)` lower-triangular
decay matrix.

So the rows and columns of `decay_chunk` are both `c+1`, and **the last row / last column
corresponds to "after all chunks"**.

### ★ Point 2.2: The Matrix Must Not Be Transposed

Upstream `out = torch.einsum("bhzc,bchd->bzhd", decay_chunk, states)`:

- `decay_chunk[b][h][z][c]` — the **row `z` is the output chunk, the column `c` is the
  summed chunk**;
- `out[b][z][h][d] = Σ_c decay_chunk[b][h][z][c] * states[b][c][h][d]`.

Transposing that matrix changes the result (the decay then flows backwards instead of
forwards).

### ★ Point 2.3: `decay_chunk` Is Lower-Triangular (Including the Diagonal), and the Diagonal
Is Always `1.0`

`masked_fill(~tril(...), 0)` keeps only the lower triangle including the diagonal.
The diagonal is `exp(0) = 1.0`.

### ★ Point 2.4: How `passed` and `final_state` Are Split

Upstream `return out[:, :-1], out[:, -1]` — `passed` is the **first `c`** entries and
`final_state` is the **last one**. The length of `passed` is `c`, not `c+1`.

## mode 3: `chunk_scan` —— The Intra-Chunk Output Ledger

```python
solve("chunk_scan", B_mat=..., C_mat=..., x=..., dt=..., dA_cumsum=...,
      prev_states=..., D=None, z=None)
```

Outputs:

```json
{
  "CB":           [[[[[...]]]]],
  "decay":        [[[[[...]]]]],
  "scores_decay": [[[[[...]]]]],
  "out":          [[[[...]]]]
}
```

- `dt`: `[b][h][c][lc]`; `dA_cumsum`: `[b][h][c][lc]`.
- `prev_states`: `[b][c][h][p][n]`.
- `CB[b][c][h][l][s]`, `decay[b][h][c][l][s]`, `scores_decay[b][c][h][l][s]`.
- `out[b][l][h][p]`.

### ★ Point 3.1: `CB` Is the Intra-Chunk Pairwise Cross Term

Upstream:

```python
CB = torch.einsum("bclhn,bcshn->bchls",
                   rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                   rearrange(B, "b (c s) h n -> b c s h n", c=nchunks))
```

that is `CB[b][c][h][l][s] = Σ_n C[b][c*lc+l][h][n] * B[b][c*lc+s][h][n]`.
Both `C` and `B` are **first expanded group → head** before taking part in the computation.

### ★ Point 3.2: The Direction of `decay[l][s]` Is "Current Minus Reference"

Upstream `dt_segment_sum = dA_cumsum[:,:,:,:,None] - dA_cumsum[:,:,:,None,:]`,
so `decay[b][h][c][l][s] = exp(dA_cumsum[l] - dA_cumsum[s])`.

With `dA < 0`, when `l >= s` we get `dA_cumsum[l] <= dA_cumsum[s]` ⇒ `decay <= 1`
(decay); when `l < s` the exponent is positive ⇒ `decay > 1`, but the lower-triangular mask
then zeroes it ⇒ **causal**.

### ★ Point 3.3: `scores_decay` Is Lower-Triangular (Including the Diagonal)

`masked_fill(~tril(chunk_size, chunk_size), 0)` ⇒ the entries with `s > l` are `0`.

### ★ Point 3.4: The Output Is the "Cross Term + Initial-State Term"

```python
out = torch.einsum('bchls,bhcs,bcshp->bclhp', scores_decay, dt, x)
state_decay_out = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
out_prev = torch.einsum('bclhn,bchpn->bclhp', C, prev_states) * state_decay_out
out = out + out_prev
```

Note that `state_decay_out` is the **position-wise** `exp(dA_cumsum[l])` **inside the
chunk**, not the chunk-final cumulative value.

### ★ Point 3.5: `D` Is an **Additive** Skip Connection, `z` Is a **Multiplicative** Gate

```python
if D is not None:
    if D.dim() == 1:
        D = rearrange(D, "h -> h 1")
    out = out + x * D
return out if z is None else out * F.silu(z)
```

- a one-dimensional `D` (`[h]`) is reshaped into `[h][1]` and broadcast over `headdim`;
- `z` is an element-wise `silu` gate (**multiplied**), not added;
- `silu(z) = z * sigmoid(z)`, **not** `sigmoid(z)` itself.

## mode 4: `ssd_forward` —— The Full Three-Stage Pipeline

```python
solve("ssd_forward", x=..., dt=..., A=..., B_mat=..., C_mat=..., chunk_size=...,
      D=None, z=None, dt_bias=None, dt_softplus=False)
```

- `dt`: the **raw layout** `[b][l][h]`.
- `dt_softplus`: boolean, default `False`.

Outputs:

```json
{
  "out":            [[[[...]]]],
  "dA_cumsum":      [[[[...]]]],
  "decay_states":   [[[[...]]]],
  "states":         [[[[[...]]]]],
  "prev_states":    [[[[[...]]]]],
  "decay_chunk":    [[[[...]]]],
  "final_state":    [[[...]]],
  "CB":             [[[[[...]]]]],
  "scores_decay":   [[[[[...]]]]],
  "nchunks": 2, "chunk_size": 4, "padded_seqlen": 8
}
```

The orchestration order copies upstream `ssd_chunk_scan_combined_ref`:

```python
dt = F.pad(dt, (0, 0, 0, chunk_size - seqlen % chunk_size))   # pad up to a chunk multiple
dt = rearrange(dt, "b (c l) h -> b h c l", l=chunk_size)
dt = dt.float()
if dt_bias is not None:
    dt = dt + rearrange(dt_bias, "h -> h 1 1")
if dt_softplus:
    dt = F.softplus(dt)
dA = dt * rearrange(A, "h -> h 1 1")
dA_cumsum = torch.cumsum(dA, dim=-1)
states = chunk_state_ref(B, x, dt, dA_cumsum)
states = state_passing_ref(rearrange(states, "... p n -> ... (p n)"),
                           dA_cumsum[:, :, :, -1])[0]
states = rearrange(states, "... (p n) -> ... p n", n=dstate)
out = chunk_scan_ref(B, C, x, dt, dA_cumsum, states, D=D, z=z)
```

### ★ Point 4.1: `dt_bias` Is Added **Before** `softplus`

The upstream order is "add the bias first, then softplus". So the effect of `dt_bias` is not
`softplus(dt + bias)` alone but `softplus(dt + bias)` and **then** the multiplication by `A`.

### ★ Point 4.2: `dt_bias` Is Independent Per Head

`rearrange(dt_bias, "h -> h 1 1")` ⇒ one scalar per head, broadcast to **every chunk and
every position** of that head.

### ★ Point 4.3: The `(headdim, dstate)` Flatten Is in C Order

Upstream `rearrange(states, "... p n -> ... (p n)")` and the inverse
`rearrange(states, "... (p n) -> ... p n", n=dstate)` are a pair, and the flat index is
`p * n + ni` (`p` is the high digit).

### ★ Point 4.4: ★ The Official Reference Is Self-Inconsistent — a Non-Multiple `seqlen`
Crashes

Upstream `ssd_chunk_scan_combined_ref` pads `dt` up to a multiple of `chunk_size`, but the
downstream `chunk_scan_ref` contains:

```python
assert seqlen == nchunks * chunk_size
```

which uses the **original `seqlen`** ⇒ when `seqlen % chunk_size != 0` it **raises
AssertionError** immediately.

In other words the official reference implementation **crashes** on that path (the Triton
kernel takes a different path and is unaffected), and the `F.pad` padding branch inside
`chunk_state_ref` **can never be reached** through it.

**This task requires you to raise `ValueError` explicitly in that situation** (the message
must make it clear that `ssd_forward` itself rejected it).

## Common Mistakes

1. Writing `decay_states` with the "accumulated from the start of the chunk" convention
   (direction reversed).
2. Not guaranteeing that the last entry of `decay_states` is `1.0` (forgetting `exp(0)`).
3. Accumulating `dA_cumsum` across chunks (upstream is chunk-local).
4. Expanding group → head in `(h g)` order (invisible when `ngroups == 1`).
5. Summing the einsum of `states` once more over the heads.
6. Sizing `decay_chunk` as `c x c` (upstream is `(c+1) x (c+1)`).
7. Transposing `decay_chunk` (the row is the output chunk, the column the summed chunk).
8. Using an upper-triangular `decay_chunk` instead of a lower-triangular one.
9. Giving `passed` length `c+1` (upstream is `c`, the last entry is `final_state`).
10. Forgetting to concatenate the initial state in front of the states in `state_passing`
    (the length becomes `c+1` after the `cat`).
11. Using the **unexpanded** `C` / `B` for `CB` (the group → head expansion is required).
12. Writing `decay[l][s]` as "reference minus current".
13. Forgetting the lower-triangular mask on `scores_decay` (future positions leak).
14. Using the chunk-final cumulative value for `state_decay_out` (it must be the
    position-wise value inside the chunk).
15. Treating `D` as a multiplication (upstream is the **additive** skip `out + x * D`).
16. Treating `z` as an addition (upstream is the **multiplicative** gate `out * silu(z)`).
17. Writing `silu(z)` as `sigmoid(z)`.
18. Adding `dt_bias` after `softplus`.
19. Treating `dt_bias` as a per-position quantity (upstream is one scalar per head).
20. Flattening `(headdim, dstate)` into `(dstate, headdim)` order.
21. Silently padding in `ssd_forward` when `seqlen` is not a multiple (upstream crashes; this
    task requires an error).
22. Believing that the `F.pad` branch of `chunk_size` takes effect when `seqlen` is not a
    multiple.

## Self-check

```python
solve("chunk_state", x=..., dt=..., A=..., B_mat=..., chunk_size=4)
```

You should get `decay_states` with the last entry of every row exactly `1.0`, `dA_cumsum`
starting over at zero for each chunk, and `expanded_B` indexed by
`head = g * (h // G) + hi`.

```python
solve("state_passing", states=..., dA_chunk_cumsum=...)
```

You should get `decay_chunk` of size `(c+1) x (c+1)`, an all-`1.0` diagonal, and
`len(passed) == c`.

```python
solve("chunk_scan", B_mat=..., C_mat=..., x=..., dt=..., dA_cumsum=...,
      prev_states=..., D=[...], z=...)
```

You should get an all-zero upper triangle in `scores_decay`, a changed `out` when `z` is set
to `None` (multiplicative gate), and a changed `out` when `D` is set to `None` (additive
skip).

```python
solve("ssd_forward", x=..., dt=..., A=..., B_mat=..., C_mat=..., chunk_size=4)
```

The intermediate quantities of the three stages (`states` / `prev_states` / `final_state`)
must agree bit-for-bit with calling `chunk_state` / `state_passing` / `chunk_scan`
separately.
