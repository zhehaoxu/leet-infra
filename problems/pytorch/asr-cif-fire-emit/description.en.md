# Paraformer CIF Predictor Fire-and-Emit Ledger

Paraformer (FunASR) skips autoregressive decoding: it uses **CIF
(Continuous Integrate-and-Fire, paper 1905.11235)** to "accumulate" acoustic
embeddings from encoder hidden states by weight — each frame carries an
emission weight alpha in `[0,1)`; when the running total crosses the
threshold 1.0 it "fires", emitting one token embedding and leaving the
overflow weight to the next token. This is the repo's only
integrate-and-fire problem. Official implementation: `cif()` and
`tail_process_fn` in FunASR `funasr/models/paraformer/cif_predictor.py`.

## Interface

```python
solve(alphas: list[float], hidden: list[list[float]], threshold: float = 1.0,
      tail_threshold: float = 0.45, use_tail: bool = True) -> dict
```

- `alphas`: per-frame emission weights (officially sigmoid outputs after
  smoothing/noise gating, valid range `[0,1)`); `hidden`: `(T, D)` encoder
  states with the same frame count as `alphas` (official invariant
  `len(hidden) == len(alphas)`);
- return `{"embeds": emitted embeddings, "fire_positions": [firing frames],
  "token_num": int}`.

## State machine (faithful to official `cif()`, single-sequence view)

For each frame `t`, `alpha = alphas[t]`:

1. `distribution_completion = 1 - integrate` — taken **before** adding
   alpha; the weight still missing for the current token;
2. `integrate += alpha`;
3. **snapshot of the pre-fire value** (official `list_fires.append` runs
   before the decrement);
4. `fire = integrate >= threshold` — **closed interval**, exactly reaching
   the threshold also fires;
5. on fire, `integrate -= 1` and `cur = completion` (the gap-filling part
   of this alpha); otherwise `cur = alpha`; `remainds = alpha - cur`
   (overflow weight);
6. `frame += cur * hidden_t`; on fire **emit frame** (the accumulated
   weighted sum of the current token) and reset `frame` to
   `remainds * hidden_t`.

The emitted embeddings are the frame snapshots at the firing frames;
`fire_positions` is ascending.

## The two official token_num conventions

- `use_tail=True` (official inference path): `tail_process_fn` **appends
  one frame `alpha = tail_threshold`** to `alphas` and appends one
  **zero-vector row** to `hidden` — the tail frame pairs with that zero row
  (official invariant `len(hidden) == len(alphas)`), so the gap-filling
  weight multiplies a zero vector and the final token's embedding comes
  only from the real frames. `token_num = floor(total weight after the
  append)`. For a single sequence the emission count always equals that
  floor (provable: after every fire the remainder is < 1; appending tail
  either exactly completes one more token or stays below 1).
- `use_tail=False` (the `len_labels` convention inside official `cif()`):
  `token_num = round(total weight)`. Here the emission count is determined
  by the snapshots and **may differ from token_num** (tail remainder is
  dropped).

## Examples

```python
h = [[1,0],[0,1],[2,2],[3,3]]
solve([0.5,0.5,0.5,0.5], h, use_tail=False)
# fires at 1 and 3; embeds=[[0.5,0.5],[2.5,2.5]]; token_num=round(2.0)=2

solve([0.6,0.6,0.6], h[:3], use_tail=False)
# fires only at 1; embeds=[[0.6,0.4]] (0.2 remainder dropped);
# token_num=round(1.8)=2  — emission count 1 ≠ token_num 2

solve([0.6,0.6,0.6], h[:3], use_tail=True)
# tail frame pairs with the zero row; fires at 1 and 3(0.8+0.45=1.25)
# embeds=[[0.6,0.4],[1.2,1.4]] (2nd = 0.2h1+0.6h2; zero row contributes 0)
# token_num=floor(2.25)=2

solve([0.55], [[1.0,2.0]], use_tail=True)
# remainder 0.55 + tail 0.45 = 1.0 exactly fires -> fire at 1
# embeds=[[0.55,1.1]]; token_num=floor(1.0)=1
```

## Hints

- On fire `cur` is the **gap**, not the whole alpha;
  `remainds = c_t - k` (the overflow after the cumsum crosses integer k)
  becomes the starting weight of the next token.
- A wrong snapshot timing (post-decrement value) loses emissions that hit
  the threshold exactly (snapshot 0.0 < 1).
- Without tail, `round` has nothing to do with the emission count — it only
  affects the official batch pad width; here you only report the value.
