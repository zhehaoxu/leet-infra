# order 844 · RetNet retention's three-form equivalence ledger: the exponential decay matrix, the recurrent state recurrence, and chunkwise cross-chunk decay

> Official source: `Jamie-Stirling/RetNet` (the reference implementation of the RetNet paper) — `src/retention.py`, `src/xpos_relative_position.py`

## Background

RetNet (Retentive Network) replaces softmax attention in linear attention with a **fixed exponential decay**.
Its core mechanism is called **retention**: the influence of position `n` on position `m` (`n >= m`) is
`γ^(n-m)` — a **constant decay kernel** that does not depend on the data.

The same retention has **three completely equivalent representations** in the official code:

| Form | Complexity | Official method |
|---|---|---|
| parallel | `O(L²)` | directly computes `D ∘ (QK^T) V` |
| recurrent | `O(L)` | state recurrence `s_n = γ s_{n-1} + K_n^T V_n` |
| chunkwise | `O(L·c)` | D inside a chunk, recurrent state across chunks |

All three forms must produce **the same output**. This is the central claim of the RetNet paper and the core of this problem.

On top of retention, RetNet layers **xPos relative positional encoding** (`XPOS`) —
a rotary-style scaling with learnable frequencies `base^(pos/512)` applied to query/key.
This problem strictly separates the **retention skeleton (pure numeric ledger)** from
**xPos (irrational powers)** into four kinds of mode, maximizing judging strength.

The official implementation has a **real equivalence bug** in the chunkwise form
(the last chunk's xPos offset is misaligned). This problem reproduces it explicitly
and distinguishes it from the correct behavior.

---

## Conventions (the single source of truth for this problem)

- **mode 1 / mode 2 are the "pure retention skeleton"**: no xPos.
  All inputs are integers or rational strings (e.g. `"1/2"`, `"9/10"`),
  and the whole chain is computed exactly with `fractions.Fraction`,
  judged by **bit-exact equality** (no tolerance).
- **mode 3 / mode 4 contain irrational quantities** (powers, trigonometric functions, `exp`/`log`/`sqrt`):
  - mode 3's `scale` / `sin` / `cos` use a **relative tolerance of `1e-9`**;
  - mode 4's `gammas` use a **relative tolerance of `1e-12`**,
    while `y_cat` / `y_gn` / `out` use **absolute tolerances** (`1e-8` / `1e-6` / `1e-6`),
    because GroupNorm's `rstd = 1/sqrt(var + eps)` amplifies the ~1e-11 upstream tail error
    to the 1e-8 range (see the "tail-error amplification chain" in notes).
- **`gamma` must lie strictly inside `(0, 1)`** (given as a rational string).
  The official `_get_D` evaluates `γ^(negative)` when `n < m`, which may overflow to `inf`,
  and `inf * 0 = NaN`; the official code patches it back with `D[D != D] = 0`.
  **This problem only follows the semantics "`n < m` is always 0" and never enters the NaN branch.**

---

## mode 1 · `decay_matrix`

The decay matrix and the power table of γ from the official `SimpleRetention._get_D`.

**Arguments**

| Parameter | Type | Description |
|---|---|---|
| `gamma` | str | rational string, e.g. `"1/2"`, `"9/10"`, `"2/3"` |
| `seq_len` | int | sequence length `L`, `>= 1` |

**Returns**

```json
{
  "gamma_powers": [...],   // γ^k, k = 0..seq_len-1
  "D": [[...]]             // the L x L decay matrix
}
```

**Semantics**

- `gamma_powers[k] = γ^k`, with `gamma_powers[0] = 1`.
- `D[n][m] = γ^(n-m)` when `n >= m`; **positions with `n < m` are always `0`**.

**Illegal inputs must raise `ValueError`**: `gamma` outside `(0, 1)`, `seq_len <= 0`.

---

## mode 2 · `retention_forms`

The equivalence ledger of the three representations. **This is the core of the problem.**

**Arguments**

`Q`, `K`, `V` are all **lists grouped by head**: the outer length is the head count `H`,
and each element is an `L`-row matrix (`Q[h][n]` is the vector of head `h` at position `n`).

| Parameter | Type | Description |
|---|---|---|
| `Q` | `[[[[int]]]]` | `H` groups, each with `L` rows, row width = `head_dim` |
| `K` | `[[[[int]]]]` | same, row width = `head_dim` |
| `V` | `[[[[int]]]]` | same, **row width = `v_dim` (independent of the Q/K width)** |
| `gamma` | str | rational string inside `(0, 1)` |
| `chunk_size` | int | chunk length `c`, `>= 1` (defaults to `L`) |

**Returns**

```json
{
  "parallel":    [[[...]]],   // (L, heads, v_dim)
  "recurrent":   [[[...]]],   // (L, heads, v_dim)
  "chunkwise":   [[[...]]],   // (L, heads, v_dim)
  "scores":      [[[...]]],   // (heads, L, L)
  "states":      [[[[...]]]], // (L, heads, head_dim, v_dim)
  "block_states": [[[[...]]]] // (num_chunks, heads, head_dim, v_dim)
}
```

**Semantics of the three forms**

1. **parallel**:
   ```
   scores[h][n][m] = D[n][m] * sum_d Q[h][n][d] * K[h][m][d]
   parallel[n][h][e] = sum_m scores[h][n][m] * V[h][m][e]
   ```
   Note the output is **kept per head**: `parallel[n][h]` is the `v_dim`-dimensional vector
   of head `h` at position `n`. All heads share the same `v_dim`
   (the official `double_v_dim=False` gives `v_dim = head_size`;
   this problem allows the width of `V` to differ from the width of `Q`/`K`).

2. **recurrent**:
   ```
   s_n = γ * s_{n-1} + K_n^T V_n          (s_{-1} = 0)
   recurrent[n][h][e] = sum_d Q[h][n][d] * s_n[h][d][e]
   ```
   **In the recurrent form `K_n` and `V_n` hold only the current step**
   (shapes `(1, head_dim)` and `(1, v_dim)`), so `K_n^T V_n` is a **single-step outer product**:
   ```
   (K_n^T V_n)[d][e] = K[h][n][d] * V[h][n][e]
   ```
   There is **no summation over `m`**. Writing `sum_m` is the most common mistake.

   `states[n][h][d][e]` is exactly entry `(d, e)` of `s_n[h]`.

3. **chunkwise**: the sequence is cut into `ceil(L / c)` chunks; chunk `bi` covers
   `[bi*c, min((bi+1)*c, L))` and has length `c' = min(c, L - bi*c)`.
   ```
   D_c        = the decay matrix (c' x c', using the same γ)
   r_i        = (K_blk^T @ (V_blk * D_c[-1])) + (γ^c') * r_{i-1}
   inner      = ((Q_blk @ K_blk^T) * D_c) @ V_blk
   e[j]       = γ^(j+1)                              (j = 0..c'-1)
   cross      = (Q_blk @ r_{i-1}) * e
   output     = inner + cross
   ```
   Four details:
   - **`D_c[-1]` is the last row of the decay matrix** = `[γ^(c'-1), ..., γ^0]`,
     used to weight `V` position by position.
   - **The decay of `r_i` is `γ^(c')`** (`c'` = **the actual length of this chunk**).
     The local variable `chunk_size = x_i.shape[1]` inside the official
     `forward_chunkwise` **is the actual length of this chunk**, so for a short final
     chunk the exponent is `γ^(c')` rather than `γ^c`.
     This difference is **not observable in the output** (the final chunk's `r_i` is
     never used again), so this problem additionally returns `block_states`
     (the `r_i` at the end of each chunk) to expose and judge it explicitly.
   - **`cross` uses the [old state `r_{i-1}`]**, not the freshly computed `r_i`.
     The official source writes `cross_chunk = (Q @ r_i_1) * e` with exactly that old state.
   - **`e[j] = γ^(j+1)`** (not `γ^j`).

   `block_states[bi][h][d][e]` is entry `(d, e)` of that head's state `r_i` at the end
   of chunk `bi` (the official `forward_chunkwise` also returns `r_i`).

   **This problem's convention**: the chunkwise output **must be bit-exactly equal to the
   parallel output** (whether or not `L` is divisible by `c`).
   See the "official chunkwise last-chunk quirk" below for the reason.

**Equivalence assertions (checked by the judge)**

```
parallel == recurrent == chunkwise        (bit-exact)
```

**The official chunkwise last-chunk quirk**

The third parameter `i` of the official `forward_chunkwise(x_i, r_i_1, i)` is the **chunk index**,
and inside the function `offset = i * chunk_size`, where `chunk_size` is taken from
**`x_i.shape[1]`, i.e. the actual length of this chunk**.

- When `L % c == 0`, every chunk has length `c`, so `offset = chunk_index * c`
  is the true starting absolute position ✓.
- **When `L % c != 0`, the last chunk has length `c' < c`**, yet the official code still
  computes `offset = chunk_index * c'`, which is **not the true absolute starting position**
  (the true one is `chunk_index * c`).

Measured (L=7, c=3): the last chunk (index 2, length 1) gets offset `2 * 1 = 2`
while the true absolute position is `6` → **the last chunk's output differs by 8.78e-02**
(the first 6 positions differ by at most 4.5e-8, pure floating-point tail error).

**Controlled experiment (separating the two variables)**:

| Setting | max difference against parallel |
|---|---|
| fix the xPos offset to the chunk's true absolute start | `4.47e-08` (pure tail error) |
| fix the xPos offset to 0 everywhere | `5.07e-01` (semantically wrong) |

⇒ **The only root cause is the xPos offset**, unrelated to `γ^c` versus `γ^(c')`.

**This problem's convention**: mode 2 is the **pure skeleton** (no xPos), so the offset never
enters the computation, and chunkwise is **bit-exactly equal** to parallel regardless of
whether `L` is divisible by `c`.

**Illegal inputs must raise `ValueError`**: empty `Q`/`K`/`V`, mismatched head counts,
mismatched sequence lengths, empty row width, `gamma` out of range, `chunk_size <= 0`.

---

## mode 3 · `xpos_scale`

The scale / sin / cos ledger of the `XPOS` relative position encoding.

**Arguments**

| Parameter | Type | Description |
|---|---|---|
| `head_dim` | int | head dimension, **must be even** (`>= 2`) |
| `seq_len` | int | sequence length `L`, `>= 1` |
| `offset` | int | position offset, `>= 0` (default 0) |
| `downscale` | bool | whether to apply inverse scaling to the key (default `False`) |

**Returns**

```json
{
  "base":          [...],    // head_dim / 2 entries
  "scale":         [[...]],  // (L, head_dim/2)
  "sin":           [[...]],  // (L, head_dim/2)
  "cos":           [[...]],  // (L, head_dim/2)
  "applied_scale": [[...]]   // the scale actually multiplied onto sin/cos
}
```

**Semantics**

1. **base**:
   ```
   base[j] = (2j + 0.4 * head_dim) / (1.4 * head_dim)      j = 0..head_dim/2-1
   ```
   The official `scale` buffer is
   `(torch.arange(0, head_dim, 2) + 0.4*head_dim) / (1.4*head_dim)`,
   and it **is float32** (an int tensor combined with a Python float is promoted to float32).
   This problem reproduces that base at **float32 precision**.

2. **scale**:
   ```
   total = seq_len + offset
   scale[pos][j] = base[j] ** (pos / 512)         pos = 0..total-1
   ```
   with `scale_base = 512`.

3. **Truncation**: when `total > seq_len` (that is, when `offset > 0`),
   **`scale`, `sin` and `cos` are truncated together to the last `seq_len` rows**.

4. **sin / cos** (`fixed_pos_embedding`):
   ```
   dim = head_dim / 2                      ← the column count of scale, not head_dim
   inv_freq[j] = 1 / (10000 ** (j / dim))
   angle[pos][j] = pos * inv_freq[j]
   sin[pos][j] = sin(angle[pos][j]) ;  cos likewise
   ```
   The official `inv_freq` is **float32** (`torch.arange(0, dim)` has no dtype → float32),
   so `inv_freq[1] = 0.10000000149011612` instead of exactly `0.1`.
   This problem reproduces `inv_freq` and the angle product at float32.

5. **Official ordering**: **sin/cos are computed on the full `total` rows of `scale` first,
   and only then are the three truncated together**.
   Truncating the scale before computing sin/cos is wrong.

6. **downscale**:
   ```
   applied_scale = 1 / scale            (when downscale=True)
   applied_scale = scale                (when downscale=False)
   ```
   **The inversion happens after `fixed_pos_embedding`**, so **sin/cos are computed from the
   non-inverted scale** and downscale **does not change sin/cos**.
   Internally the official `apply_rotary_pos_emb` does `duplicate_interleave(sin * scale)`,
   so `applied_scale` is the scale actually multiplied onto sin/cos.

**Illegal inputs must raise `ValueError`**: odd or non-positive `head_dim`,
`seq_len <= 0`, `offset < 0`.

---

## mode 4 · `multiscale_retention`

The complete ledger of the official `MultiScaleRetention`: multi-scale γ + per-head retention
+ GroupNorm + swish gating.

**Arguments**

| Parameter | Type | Description |
|---|---|---|
| `X` | `[[float]]` | `(L, hidden)` |
| `W_Q` / `W_K` | `[[[[float]]]]` | one `(hidden, head_size)` matrix per head, `heads` groups |
| `W_V` | `[[[[float]]]]` | one `(hidden, head_size)` matrix per head, `heads` groups |
| `W_G` | `[[float]]` | `(hidden, v_dim)`, the gating projection |
| `W_O` | `[[float]]` | `(v_dim, hidden)`, the output projection |
| `heads` | int | head count, `>= 1`; `hidden % heads == 0` |
| `chunk_size` | int | chunk length (optional, defaults to `L`) |

**Returns**

```json
{
  "gammas": [...],    // heads entries
  "y_cat":  [[...]],  // (L, hidden)  per-head outputs concatenated
  "y_gn":   [[...]],  // (L, hidden)  after GroupNorm
  "out":    [[...]]   // (L, hidden)  the final output
}
```

**Semantics**

1. **γ sequence**:
   ```
   γ_h = 1 - exp(linspace(log(1/32), log(1/512), heads)[h])
   ```
   The first entry is exactly `1 - 1/32 = 0.96875` and the last is exactly
   `1 - 1/512 = 0.998046875`; the middle entries involve irrational exponents.
   When `heads == 1` the value is `1 - 1/32`.

2. **Per-head retention**: each head runs the parallel-form retention with its own `γ_h`,
   **and xPos must be included** (the official `SimpleRetention.forward` applies
   `Q = self.xpos(Q)` and `K = self.xpos(K, downscale=True)`).
   **xPos is a required part of retention**: measured against the official
   `SimpleRetention.forward`, the "pure decay skeleton" differs by `0.0515`,
   while adding xPos reduces the difference to `0.0`.

3. **cat**: `y_cat[n]` concatenates the `heads` per-head outputs (each `head_size` wide),
   for a total width of `heads * head_size = hidden` (`v_dim = hidden`).

4. **GroupNorm** (`nn.GroupNorm(heads, v_dim)`, `affine=True` → `weight=1`, `bias=0`, `eps = 1e-5`):
   - The official code applies it to `Y.reshape(-1, v_dim)` → **independently per position**
     (the batch and sequence dimensions are flattened together into the row dimension).
   - Each group covers `v_dim / heads` **contiguous channels** (sliced in channel order, not interleaved).
   - **The variance is biased**: `var = sum((x - mean)^2) / N` (denominator `N`, not `N-1`).
   - `y_gn = (x - mean) / sqrt(var + eps)`.
   - The variance after normalization is always `raw_var / (raw_var + eps)`.

5. **Gating and output**:
   ```
   out = (swish(X @ W_G) * y_gn) @ W_O
   swish(x) = x * sigmoid(x)
   ```

**Illegal inputs must raise `ValueError`**: empty `X`, `heads <= 0`, `hidden % heads != 0`,
an **odd** `head_size` (the official `XPOS(head_size)`'s `rotate_every_two` crashes outright on
an odd head dimension because `torch.stack` requires equal sizes — the zero-padding branch in
the official source sits *after* the `stack` and is therefore **dead code**),
mismatched head counts or row counts in `W_Q`/`W_K`/`W_V`.

---

## Common mistake checklist

| Mistake | Consequence |
|---|---|
| computing `D[n][m]` as `γ^(n-m)` when `n < m` | negative exponent overflows to `inf`/`NaN`, wrong semantics |
| writing the recurrent `K_n^T V_n` as `sum_m` | the whole state and output are wrong |
| using the whole-sequence `K`/`V` for `K_n^T V_n` | same as above |
| letting chunkwise `cross` use the freshly computed `r_i` instead of the old `r_{i-1}` | cross-chunk contribution counted twice |
| decaying `r_i` with `γ^c` (fixed chunk length) instead of `γ^(c')` (the official semantics: actual chunk length) | the last chunk of `block_states` is wrong |
| writing chunkwise `e[j]` as `γ^j` | cross-chunk decay is off by one |
| taking `D_c[-1]` as the first row | the in-chunk weighting is inverted |
| recomputing sin/cos **after** inverting the scale for downscale | the downscale semantics are wrong |
| computing sin/cos from the already-truncated scale | every position is wrong |
| using an exact `1/10000^(j/dim)` for `inv_freq` instead of float32 | the tail error exceeds `1e-9` |
| using `head_dim` instead of `head_dim/2` as the `dim` of `fixed_pos_embedding` | sin/cos are entirely wrong |
| implementing `duplicate_interleave` as "duplicate the row then concatenate" | the interleaving order is wrong |
| implementing `rotate_every_two` as concatenation instead of interleaving | the rotation signs are wrong |
| allowing an odd `head_dim` / `head_size` | the official code crashes, so an error must be raised |
| omitting xPos in the multi-scale mode | the output differs by `0.0515` |
| using the unbiased variance (`N-1`) in GroupNorm | `y_gn` is too small |
| computing GroupNorm statistics across positions instead of per position | `y_gn` is entirely wrong |
| grouping channels by **interleaving** in GroupNorm | `y_gn` is entirely wrong |
| writing `swish` as `sigmoid(x)` | the gating is wrong |
| using `linspace(1/32, 1/512)` instead of `linspace(log(1/32), log(1/512))` for the γ sequence | the whole γ sequence is wrong |
| computing mode 3 / mode 4 entirely with exact rationals | irrational quantities get truncated and the tolerance no longer suffices |

---

## Self check

```python
# mode 1: gamma = 1/2, L = 3
# gamma_powers = [1, 1/2, 1/4]
# D = [[1,   0,   0  ],
#      [1/2, 1,   0  ],
#      [1/4, 1/2, 1  ]]

# mode 2: single head, L=2, gamma=1/2
# Q = [[1,2],[3,4]], K = [[1,0],[0,1]], V = [[1,0],[1,1]]
# scores[0][1][0] = 1/2 * 3 = 3/2 ;  scores[0][1][1] = 1 * 4 = 4
# parallel[1][0]   = (3/2)*[1,0] + 4*[1,1] = [11/2, 4]
# recurrent[1][0] is bit-exactly equal to parallel[1][0]
# chunkwise[1][0] is bit-exactly equal to parallel[1][0] (for c=1 or c=2)
# block_states[0][0] = K_0^T V_0 (single-step outer product), same as states[0][0] (c=1)

# mode 3: head_dim=4
# base = [1.6/5.6, 3.6/5.6] = [2/7, 9/14]
# scale[0] is all 1s (base ** 0)
# sin[0] is all 0s ; cos[0] is all 1s

# mode 4: GroupNorm with num_groups=1 normalizes the whole row to zero mean
# the variance after normalization = raw_var / (raw_var + 1e-5)
```
