# KV Cache: CapKV's Information-Theoretic Leverage-Score Token Selection

## Problem

KV cache compression usually lets attention scores do the judging. CapKV (arXiv 2604.25975) uses a different criterion: if a token's value can be explained by a **linear combination of the other values**, the token is redundant and keeping it just repeats information that is already there. The quantity that measures "how hard to explain" is the **leverage score** of the value matrix. It takes two steps.

**Step one: build a query anchor out of the future RoPE.** Real scoring happens at later decode steps, so first average the rotation matrices of the next `n_future_positions=512` positions and apply the average to the mean query (this problem takes the `(cos, sin)` tables as explicit inputs, any table length works):

\[
\bar q=\Bigl(\tfrac{1}{L_q}\sum_i q_i\Bigr)\bar R^{\top},\qquad
\bar R=\tfrac{1}{P}\sum_{p=1}^{P}R_p,\qquad R_p=\operatorname{diag}(\cos_p)+\sin_p\cdot \mathrm{rot}
\]

Rotating every token and then averaging is exactly the same as averaging first and rotating once (matrix multiplication is linear); upstream writes it in the second order.

**Step two: the weighted leverage score.** Compute a cosine statistic between every key and the anchor, then turn it into exponential weights:

\[
s_j=\operatorname{clamp}\bigl(\cos(\bar q,k_j)\bigr),\qquad
w_j=\exp\bigl(\tau(s_j-\max_{j'}s_{j'})\bigr),\quad \tau=5
\]

Next, build a capacity matrix out of the values and take the quadratic form of each value under its inverse:

\[
A=I+\sum_j (\sqrt{w_j+\epsilon}\,v_j)(\sqrt{w_j+\epsilon}\,v_j)^{\top},\qquad
\ell_j=v_j^{\top}A^{-1}v_j
\]

The final score is `w_j * ℓ_j`: a token with a large weight that genuinely adds an independent direction scores high; a token with a large weight whose value is collinear with the others gets a small `ℓ_j`. Three details that are easy to get wrong:

1. **The first 4 tokens are sinks**: they take no part in the statistic, the weights or `A`, and get no leverage — yet they are **always kept**, with their score filled with the global maximum + 1;
2. **average inside each GQA group**: `repeat_kv` lays the same kv head out `group_size` times in a row, so you average over contiguous blocks of `group_size`, not take the first copy;
3. `epsilon=1e-6` goes inside `sqrt(w+epsilon)` (it regularises the weights and keeps `A` strictly positive definite), **not** on the diagonal of `A`.

## Task

Implement `solve(q, k, v, cos, sin, keep, group_size=1)`:

- `q`: `(Lq, Hq, D)` with `Hq = Hkv * group_size`; `k`, `v`: `(T, Hkv, D)`, same shape, same dtype as `q` (float32/float64); `cos`, `sin`: RoPE tables of shape `(P, D)`, `P >= 1`, `D` a positive even number; `keep`: an integer in `[4, T]`, `group_size` a positive integer.

All arithmetic inside is float64, results come back in the input dtype (except `kept`), and inputs must not be mutated. Return a 5-key dict; write `T' = T - 4`:

- `statistic`, `weights`, `leverage`: `(Hq, T')`, covering only the tokens after the sinks;
- `scores`: `(Hkv, T)`, whose first 4 columns hold the sink value `scores.max() + 1` (a global max across heads);
- `kept`: `(Hkv, keep)` int64, each row takes the `keep` best scores in descending order and is then returned ascending, ties going to the smaller token index.

Raise `ValueError` on: not 3-D, mismatched dtype or a dtype other than float32/float64, `k`/`v` shape mismatch, `D` not a positive even number, `Lq < 1`, `group_size` not a positive integer, `Hq != Hkv * group_size`, `T <= 4`, `cos`/`sin` not same-shaped `(P, D)`, `keep` not an integer in `[4, T]`.

## Example

```
q  = [[[1.0, 0.0]], [[0.0, 1.0]]]                        # (Lq=2, Hq=1, D=2)
k  = [[[9.0, 9.0]]] * 4 + [[[1.0, 0.0]], [[0.0, 1.0]]]    # (T=6, Hkv=1, D=2)
v  = [[[0.0, 0.0]]] * 4 + [[[1.0, 0.0]], [[0.0, 1.0]]]
cos = [[1.0, 1.0], [0.0, 0.0]]                           # P=2 future positions
sin = [[0.0, 0.0], [1.0, 1.0]]
keep = 5

mean(R)   = (I + rot) / 2 = [[0.5, -0.5], [0.5, 0.5]]    # mean(q) = [0.5, 0.5], rot = [[0, -1], [1, 0]]
anchor    = [0.5, 0.5] @ mean(R)^T = [0.0, 0.5] -> normalize -> [0.0, 1.0]
statistic = [[0.0, 1.0]]                                 # cosine with e1 / e2
weights   = [[0.0067379, 1.0]]                           # [exp(-5), 1]
leverage  = [[0.9933062, 0.49999975]]                    # 1 / (1 + w_i + eps)
scores    = [[1.49999975] * 4 + [0.0066928, 0.49999975]]  # w_i * leverage_i, first 4 slots = max + 1
kept      = [[0, 1, 2, 3, 5]]                            # top 5 descending, then ascending
```

## Constraints

- **normalize eps is 1e-12**: `k_j = 0` or a degenerate anchor gives a cosine of 0 rather than NaN; the statistic is then clamped to `[-1, 1]`. **After the max shift the weights peak at exactly 1**: `exp(τ(s - max s))` cannot overflow, and it also guarantees the sink's `max + 1` beats every real score.
- **the `w` inside `sqrt(w + eps)` is the unscaled weight**: the capacity matrix is assembled from `√(w+eps)·v`, while the leverage quadratic form uses the **original** `v`; get either one wrong and the score is no longer CapKV's score.
- **`A` is positive definite by construction**: `A = I + a PSD term`, so call `linalg.solve` directly — no inverse, no diagonal jitter; when `v_j = 0` you get `ℓ_j = 0` and a zero score instead of a NaN.
- **the sink fill is the global max + 1**: `scores.max()` runs across all kv heads, not each head's own max + 1; with several heads the two differ on any head whose scores are all 0.
- **ties go to the smaller token index**: the descending `argsort` must be stable; `torch.topk` does not define tie order, so this problem pins it.
