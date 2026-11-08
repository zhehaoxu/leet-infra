# order 844 · Source dissection: RetNet retention's three forms and xPos

> Official source: `Jamie-Stirling/RetNet` (GitHub repository id `668291913`)
> Files fetched: `src/retention.py` (204 lines), `src/xpos_relative_position.py` (94 lines),
> `src/retnet.py` (76 lines)

---

## 1. The three retention forms in the official code

`SimpleRetention` is the core module of RetNet, one instance per head.
It has three `forward*` methods that **must produce the same output** (the paper's central claim).

### 1.1 The parallel form

```python
def forward(self, X):
    sequence_length = X.shape[1]
    D = self._get_D(sequence_length).to(self.W_Q.device)

    Q = (X @ self.W_Q)
    K = (X @ self.W_K)

    Q = self.xpos(Q)
    K = self.xpos(K, downscale=True)

    V = X @ self.W_V
    ret = (Q @ K.permute(0, 2, 1)) * D.unsqueeze(0)

    return ret @ V
```

Shape flow (`B` = batch, `H` = head, `L` = seq, `d` = head_size, `v` = v_dim):

- `X`: `(B, L, hidden)`
- `Q`, `K`: `(B, L, d)`
- `V`: `(B, L, v)`
- `Q @ K.permute(0,2,1)`: `K.permute` turns `(B, L, d)` into `(B, d, L)`,
  which broadcasts into the `(B, L, L)` matrix of `q_n · k_m`
- `D` is `(L, L)` → `D.unsqueeze(0)` becomes `(1, L, L)`, **broadcast per head**
- `ret @ V`: `(B, L, L) @ (B, L, v)` → `(B, L, v)`

### 1.2 The decay matrix of `_get_D` and its NaN quirk

```python
def _get_D(self, sequence_length):
    n = torch.arange(sequence_length).unsqueeze(1)
    m = torch.arange(sequence_length).unsqueeze(0)

    # Broadcast self.gamma ** (n - m) with appropriate masking to set values where n < m to 0
    D = (self.gamma ** (n - m)) * (n >= m).float()  #this results in some NaN when n is much larger than m
    # fill the NaN with 0
    D[D != D] = 0

    return D
```

- `D[n][m] = γ^(n-m)` when `n >= m`; positions with `n < m` are wiped to 0 by the `(n >= m)` factor.
- **Official quirk**: when `n < m` the exponent is **negative**, so `γ^(-k)` is mathematically
  not 0 but gets wiped by the `(n>=m)` factor; when `γ` is a tiny positive number and `n-m`
  is very negative, `γ^(n-m)` can **overflow to `inf`**, and `inf * 0 = NaN`,
  hence the official `D[D != D] = 0` that patches NaN back to 0.
  **This problem only accepts `γ ∈ (0,1)` with a finite sequence length** and decides directly
  with **exact rationals** (positions with `n < m` are always 0), **never entering the NaN branch**.
- The official `D` always has `γ^0 = 1` on its diagonal.

### 1.3 The recurrent form

```python
def forward_recurrent(self, x_n, s_n_1, n):
    """
    x_n: (batch_size, 1, hidden_size)
    s_n_1: (batch_size, hidden_size, v_dim)
    """
    Q = (x_n @ self.W_Q)
    K = (x_n @ self.W_K)

    Q = self.xpos(Q, n+1)
    K = self.xpos(K, n+1, downscale=True)

    V = x_n @ self.W_V

    s_n = self.gamma * s_n_1 + (K.transpose(-1, -2) @ V)

    return (Q @ s_n), s_n
```

- The state `s_n` has shape `(B, head_size, v_dim)` (the official docstring says
  `(batch, hidden_size, v_dim)`, but it is actually `head_size`).
- **The xPos offset is `n+1`** (not `n`) — an off-by-one detail in the official code.
  That is inconsistent with the parallel form's `xpos(Q)` (offset 0), so the official recurrent
  and parallel forms are **not strictly equal once xPos is involved**; mode 2 of this problem is
  the **pure skeleton** (no xPos), so the offset never enters the computation.
- The decay is `γ * s_{n-1}` (one factor of γ per step).
- **The trap this problem is built around**: in the recurrent form `x_n` holds **only the
  current step**, so `K` is `(B, 1, d)` and `V` is `(B, 1, v)`,
  and `K.transpose(-1,-2) @ V` is `(B, d, 1) @ (B, 1, v)` — a **single-step outer product**
  with **no summation over `m`**. Writing `sum_m K[m][d] * V[m][e]` is wrong
  (my own first implementation hit exactly this and was caught by the official cross-check).

### 1.4 The chunkwise form

```python
def forward_chunkwise(self, x_i, r_i_1, i):
    """
    x_i: (batch_size, chunk_size, hidden_size)
    r_i_1: (batch_size, hidden_size, v_dim)
    """
    batch, chunk_size, _ = x_i.shape
    D = self._get_D(chunk_size)

    Q = (x_i @ self.W_Q)
    K = (x_i @ self.W_K)

    Q = self.xpos(Q, i * chunk_size)
    K = self.xpos(K, i * chunk_size, downscale=True)

    V = x_i @ self.W_V

    r_i =(K.transpose(-1, -2) @ (V * D[-1].view(1, chunk_size, 1))) + (self.gamma ** chunk_size) * r_i_1

    inner_chunk = ((Q @ K.transpose(-1, -2)) * D.unsqueeze(0)) @ V

    #e[i,j] = gamma ** (i+1)
    e = torch.zeros(batch, chunk_size, 1)

    for _i in range(chunk_size):
        e[:, _i, :] = self.gamma ** (_i + 1)

    cross_chunk = (Q @ r_i_1) * e

    return inner_chunk + cross_chunk, r_i
```

Five details:

1. **`D[-1]` is the last row of `D`** = `[γ^(c-1), γ^(c-2), ..., γ^0]` (with `c = chunk_size`),
   used to weight `V` position by position → `K^T (V * D[-1])`.
2. **The decay of `r_i` is `γ^chunk_size`**, and the local `chunk_size = x_i.shape[1]`
   inside the official function is **the actual length of this chunk** (denoted `c'`),
   so the exponent is `γ^(c')` rather than `γ^c` (the fixed chunk length).
   This difference is **not observable in the output** (the final chunk's `r_i` is
   never used again), so this problem additionally returns `block_states`
   (the `r_i` at the end of each chunk) to expose and judge it explicitly.
3. **`e[j] = γ^(j+1)`** (note `j+1`, not `j`) is the cross-chunk decay factor.
4. **`inner_chunk`** uses the **block-relative** `D` (`chunk_size × chunk_size`).
5. **The `cross_chunk = (Q @ r_i_1) * e` uses the [old state `r_i_1`]**,
   **not** the `r_i` just computed in the same function.
   My first implementation used the freshly computed `r_prev` and was caught by the official
   cross-check (the last chunk differed by 8.78e-02).
   Note that the definition of `r_i` already contains `γ^chunk_size * r_i_1`,
   so using `r_i` in the cross term would double-count the cross-chunk contribution
   and decay it twice.

---

## 2. ★★ The real quirk: official chunkwise is not equal to parallel

This is the central discovery of this problem and a **real bug** in the official implementation.

The third parameter `i` of the official `forward_chunkwise(x_i, r_i_1, i)` is the **chunk index**,
and inside the function:

```python
Q = self.xpos(Q, i * chunk_size)     # chunk_size = x_i.shape[1], the ACTUAL length of this chunk
```

- `retnet.py` calls `self.retentions[j].forward_chunkwise(x_i, r_i_1s[j], i)`,
  where `i` is the chunk index and `x_i` is a **full chunk** (length exactly `chunk_size`)
  → `offset = chunk_index * chunk_length` = **the chunk's true starting absolute position** ✓.
- **But when the sequence length is not a multiple of the chunk length, the last chunk has
  length `c' < chunk_size`**: the official code still uses `i * chunk_size`, and `chunk_size`
  there is the **actual length `c'` of the last chunk**, so the offset becomes
  `chunk_index * c'` — **not the true absolute starting position**.

Measured (L=7, c=3): the last chunk (index 2, length 1) gets offset `2 * 1 = 2`
while the true absolute position is `6` → **the last chunk's output differs by 8.78e-02**
(the first 6 positions differ by at most 4.5e-8, pure floating-point tail error).

### 2.1 A controlled experiment separating the two variables

At first I suspected `γ^chunk_size` versus `γ^(actual length)`, so I wrote `probe844d.py`
to run a **controlled experiment that separates the two variables**:

| Setting | max difference against parallel |
|---|---|
| fix the xPos offset to the chunk's true absolute start | `4.47e-08` (pure tail error) |
| fix the xPos offset to 0 everywhere | `5.07e-01` (semantically wrong) |

And: **once the offset is fixed to the absolute position, `γ**n` (actual length) and `γ**C`
(fixed chunk length) give exactly the same result.**

⇒ **The only root cause is the xPos offset**, unrelated to the decay factor.

### 2.2 This problem's convention

mode 2 is the **pure retention skeleton** (no xPos), so the offset never enters the computation,
and chunkwise is **bit-exactly equal** to parallel regardless of whether `L` is divisible by `c`.
That makes mode 2's judging the **strongest bit-exact check** (no tolerance at all), cleanly
separating "skeleton equivalence" from "positional encoding".

---

## 3. xPos relative position (`src/xpos_relative_position.py`)

### 3.1 The `XPOS` body

```python
class XPOS(nn.Module):
    def __init__(self, head_dim, scale_base=512):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )

    def forward(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = 0
        max_pos = length + offset + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x
```

Ten points:

1. **`scale_base = 512`** by default.
2. **The `scale` buffer has length `head_dim / 2`** (`arange(0, head_dim, 2)`)
   → **`head_dim` must be even**.
3. **`base[j] = (2j + 0.4*head_dim) / (1.4*head_dim)`**.
4. **`scale[pos][j] = base[j] ** (pos / 512)`**, with `pos` running from 0 to `length + offset - 1`.
5. **The truncation branch**: `scale.shape[0] = length + offset`; when it is `> length`,
   the **last `length` rows** are taken (`scale[-length:]`), and sin/cos are truncated likewise.
   → **The larger the offset, the further back the positions that get taken are.**
6. **Official ordering**: **sin/cos are computed on the full `total` rows of `scale` first,
   and only then are scale / sin / cos truncated together**.
   My first version truncated the scale before computing sin/cos → every position was wrong.
7. **The exact place of `downscale`**: `scale = 1/scale` happens **after
   `fixed_pos_embedding(scale)`**, so **sin/cos are computed from the non-inverted scale**,
   and downscale **only inverts the `scale` argument passed to `apply_rotary_pos_emb`**
   (which then multiplies it onto sin/cos).
   Measured, downscale versus not changes the output by `1.9e-02` (**it is not an identity**).
8. **The `dim` of `fixed_pos_embedding` is the column count of `scale`** = `head_dim/2`,
   **not `head_dim`**.
9. **`duplicate_interleave`**: `(m0, m1, m2) → (m0, m0, m1, m1, m2, m2)`
   (each row element is duplicated once and interleaved).
10. **`rotate_every_two`**: `x1 = x[..., ::2]`, `x2 = x[..., 1::2]`,
    `stack((-x2, x1), dim=-1).flatten(-2)` → `(a0, b0, a1, b1, ...)` (**interleaved**, not concatenated).

### 3.2 ★ An odd head_dim crashes the official code

```python
def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    if x.shape[-1]%2 == 1:
        # fill last dim with zero if hidden_size is odd
        x2 = torch.concat((x2, torch.zeros_like(x2[:, :, :1])), dim=-1)
    return x.flatten(-2)
```

`torch.stack((-x2, x1), dim=-1)` requires equal sizes, but the even indices give `ceil(d/2)`
elements and the odd indices give `floor(d/2)` → **when `d` is odd, `stack` immediately throws
`RuntimeError: stack expects each tensor to be equal size`**.

The official source's zero-padding branch sits **after** the `stack`, so it is **dead code**
(never reached). Measured: `head_dim = 5 / 7` crashes; `4 / 6` work fine.

⇒ **This problem requires `head_dim` (mode 3) and `head_size` (mode 4) to be even**,
and odd values must raise an error.

### 3.3 The float32 chain of `fixed_pos_embedding`

```python
def fixed_pos_embedding(x):
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq).to(x)
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)
```

- `torch.arange(0, dim)` has **no dtype → float32**, so `inv_freq` is float32 throughout.
  Hence `inv_freq[1] = 0.10000000149011612`, **not** exactly `0.1`.
- `torch.arange(0, seq_len, dtype=torch.float)` is float32 too; the `einsum` product
  **is also float32**, and `.to(x)` then promotes it to x's dtype (float64).
- So `angle[pos][j] = f32(f32(pos) * inv_freq[j])`, and you **cannot** use Python's
  `pos * inv_freq[j]` (a double multiplication).

### 3.4 The multiplication order of `apply_rotary_pos_emb`

```python
def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    return (x * cos[:, :x.shape[-1]]) + (rotate_every_two(x) * sin)[:, :, :x.shape[-1]]
```

- `t * scale` is an **element-wise** product (`t` and `scale` have the same `(L, dim)` shape),
  and then `duplicate_interleave` interleaves it out to `(L, 2*dim) = (L, head_dim)`.
- `cos[:, :x.shape[-1]]` truncates to `head_dim` (this only differs when `head_dim != 2*dim`;
  for even `head_dim` in this problem `2*dim == head_dim`, so they coincide).

---

## 4. `MultiScaleRetention`

```python
class MultiScaleRetention(nn.Module):
    def __init__(self, hidden_size, heads, double_v_dim=False):
        self.hidden_size = hidden_size
        self.v_dim = hidden_size * 2 if double_v_dim else hidden_size
        self.heads = heads
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        self.head_size = hidden_size // heads
        self.head_v_dim = hidden_size * 2 if double_v_dim else hidden_size

        self.gammas = (1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), heads))).detach().cpu().tolist()

        self.swish = lambda x: x * torch.sigmoid(x)
        self.W_G = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)
        self.W_O = nn.Parameter(torch.randn(self.v_dim, hidden_size) / hidden_size)
        self.group_norm = nn.GroupNorm(heads, self.v_dim)

        self.retentions = nn.ModuleList([
            SimpleRetention(self.hidden_size, gamma, self.head_size, double_v_dim) for gamma in self.gammas
        ])

    def forward(self, X):
        Y = []
        for i in range(self.heads):
            Y.append(self.retentions[i](X))

        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return (self.swish(X @ self.W_G) * Y) @ self.W_O
```

### 4.1 The γ sequence

```
γ_h = 1 - exp(linspace(log(1/32), log(1/512), heads)[h])
```

- The first entry is exactly `1 - 1/32 = 0.96875`, the last is exactly `1 - 1/512 = 0.998046875`.
- **The middle entries involve irrational exponents** (`exp` of a non-integer `log`).
- When `heads == 1`, `linspace` has a single point `log(1/32)` → γ = `0.96875`.
- **Common mistake**: writing `linspace(1/32, 1/512, heads)` (**linear** interpolation instead of
  logarithmic) → the whole sequence is wrong.

### 4.2 Four details of GroupNorm

`self.group_norm = nn.GroupNorm(heads, self.v_dim)`:

- `affine=True` (the PyTorch default) → there are learnable `weight` / `bias` parameters,
  initialized to `weight = 1`, `bias = 0` → this problem treats the affine part as the identity.
- `eps = 1e-5` (the PyTorch default).
- The official code applies it to `Y.reshape(-1, self.v_dim)` → **independently per position**
  (the batch and sequence dimensions are flattened together into the row dimension).
- Each group covers `v_dim / heads` **contiguous channels** (sliced in channel order,
  **not interleaved**).
- **The variance is biased**: `var = sum((x - mean)^2) / N` (denominator `N`, not `N-1`).
- After normalization that group's variance is always `raw_var / (raw_var + eps)`.

### 4.3 The output shape flow

```
X: (B, L, hidden)
  → per head: (B, L, head_size)          (xPos included)
  → cat dim=2: (B, L, v_dim)            v_dim = heads * head_size = hidden
  → group_norm: (B, L, v_dim)
  → (swish(X @ W_G) * Y): (B, L, v_dim)
  → @ W_O: (B, L, hidden)
```

- `swish(x) = x * sigmoid(x)`.
- ★ **xPos is a required part of retention**: measured against the official
  `SimpleRetention.forward`, the "pure decay skeleton" differs by `0.0515`;
  adding xPos reduces the difference to **0.0**.
  In other words mode 4 **must not** skip xPos.

---

## 5. The tail-error amplification chain (why mode 4's tolerances are layered)

mode 4 contains a **tail-error amplification chain** that dictates the tolerance layering:

```
xPos's base is float32          →  base carries a ~1e-8 relative tail error
  ↓ scale[pos][j] = base[j] ** (pos/512)
scale has a ~1e-11 tail error
  ↓ Q @ K^T
scores tail error ~1e-11
  ↓ ret @ V (a sum of many terms)
y_cat tail error ~1e-11
  ↓ GroupNorm: rstd = 1/sqrt(var + eps), with var ~ 8e-5
               → the relative error is amplified by 1/sqrt(8e-5) ≈ 100x
y_gn tail error ~1e-8
  ↓ (swish(X @ W_G) * Y) @ W_O — a [catastrophic cancelling sum] of 16 products
   (terms ~0.03 while the result is ~0.0038, cancelling away one order of magnitude)
out tail error amplified another 1e3x → ~5.7e-11 absolute
```

⇒ mode 4's judging must be layered:

| Quantity | Judging method |
|---|---|
| `gammas` | relative tolerance `1e-12` |
| `y_cat` | absolute tolerance `1e-8` |
| `y_gn` | absolute tolerance `1e-6` |
| `out` | absolute tolerance `1e-6` |

Meanwhile mode 1 / mode 2 are **pure rationals** judged by **bit-exact equality** (no tolerance),
and mode 3's `scale` / `sin` / `cos` use a **relative tolerance of `1e-9`**.

---

## 6. Ten reusable lessons

1. **The NaN branch of `_get_D` is not decoration**: `γ^(negative)` can overflow to `inf`,
   and `inf * 0 = NaN`. Once the problem constrains `γ ∈ (0,1)` it can be bypassed entirely,
   but you should understand why the official code writes `D[D != D] = 0`.
2. **The matrix product in the recurrent form is a single-step outer product**: `K_n` and `V_n`
   hold only the current step, and `K_n^T V_n` does not sum over `m`. This is the easiest
   place to get it wrong.
3. **Chunkwise's cross term uses the old state**: the official `cross_chunk = (Q @ r_i_1) * e`
   uses the state `r_i_1` that came in, not the `r_i` just computed.
4. **The decay exponent is the actual chunk length**: the `chunk_size` in the official
   `self.gamma ** chunk_size` is `x_i.shape[1]` (this chunk's actual length `c'`),
   not the fixed chunk length. And this difference is **not observable in the output**,
   so it can only be judged through `block_states`.
5. **`e[j] = γ^(j+1)`**: the cross-chunk decay factor carries a +1 shift.
6. **`D[-1]` is the last row**: it weights `V` position by position; taking the first row
   inverts the weighting.
7. **xPos's sin/cos are computed before the truncation**: the official order is
   "compute sin/cos on the full rows → truncate all three together".
8. **downscale comes after `fixed_pos_embedding`**: sin/cos are unchanged; only the scale
   multiplied onto sin/cos is inverted.
9. **The `dim` of `fixed_pos_embedding` is `head_dim/2`**, not `head_dim`.
   On top of that, `inv_freq` is float32 throughout, so the tail error must be reproduced in float32.
10. **Treat the official dead code as a real trap**: the zero-padding branch of
    `rotate_every_two` sits after the `stack`, so an odd `head_dim` crashes outright —
    the problem requires an even value and an explicit error.

---

## 7. The three-layer verification structure

This problem uses **three layers of verification**:

### Layer 1: official source retrieval

`src/retention.py` + `src/xpos_relative_position.py` are read line by line, and every numeric
detail (float32 precision, truncation order, the order of multiplications) is confirmed with
dedicated probe scripts.

### Layer 2: official cross-check (`/tmp/verify844.py`)

Three layers of structure:
- **The official layer** reproduces the **source expressions** of the official
  `forward` / `forward_recurrent` / `forward_chunkwise`
  (rather than calling the classes directly, because the official `SimpleRetention` projects
  Q/K/V out of the same `X` and cannot accept mutually unrelated Q/K/V).
- **The reference-solution layer** computes exactly with `Fraction`.
- **An independent oracle layer** is a separate Fraction implementation (it does not reuse the
  reference solution's code), which rules out "the reference solution and the oracle are wrong together".

Cross-check results (**all four modes green**):

| mode | max relative error |
|---|---|
| mode 1 (decay matrix) | `0.000e+00` (bit-exact) |
| mode 2 (three forms) | `1.378e-14` (pure floating-point tail error of chunkwise; `parallel == recurrent` bit-exact) |
| mode 3 (xPos) | `1.723e-16` |
| mode 4 (multi-scale) | `4.985e-08` (xPos floating-point path plus catastrophic cancellation amplification) |

### Layer 3: mutation testing (`/tmp/mut844.py`)

Single-point mutations are applied to the reference solution, and every mutation must be caught
by the judge cases (`CAUGHT`).
Design principle: **a surviving mutation is a gap in the judging**, so each survivor forces either
a new judge assertion or a correction of the mutation.

★ **A discovery about "unobservable differences"**: the mutation report showed that
changing `γ^(c')` to `γ^c` (fixed chunk length) **is not caught by the outputs** —
because the final chunk's `r_i` is never used again, so both spellings produce
**bit-identical** final outputs.
That exposed a judging gap: **the decay exponent is not observable in the output**.
The fix is to expose the per-chunk state `r_i` the way the official
`forward_chunkwise` returns it (a new `block_states` output), which makes the
exponent difference immediately judgeable.
⇒ **A surviving mutation does not always mean "the judge cases are too lax";
it can also mean "the output design missed an observable quantity".**

**Real bugs** caught in this round (each found by a surviving mutation and then traced back):

| Mutation | Consequence |
|---|---|
| `cross_chunk` uses the new `r_i` instead of the old `r_i_1` | the last chunk differs by `8.78e-02` |
| the recurrent `K_n^T V_n` written as `sum_m` | the whole state is wrong |
| the `base` formula missing the `0.4*head_dim` term in the numerator | `2/7` versus `16/5` |
| `inv_freq` using an exact `1/10000^(j/dim)` | `0.1` versus `0.10000000149011612` |
| sin/cos computed from the already-truncated scale | every position is wrong |
| `scale` kept in float32 throughout (the official base is promoted to float64) | `0.9975562` versus `0.9976` |
| the chunkwise `r_i` decay using `γ^c` (fixed chunk length) instead of the official `γ^(c')` | identical outputs (unobservable), but the last chunk of `block_states` is wrong |

**Genuinely equivalent mutations (must be removed)**:
- things like `abs(w) if w > 0 else 0` — it is equivalent to `max(w, 0)` at `w = 0`,
  so it is **truly equivalent** and not a judging gap.
