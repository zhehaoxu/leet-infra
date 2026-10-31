# KV Cache: TriAttention Phase-Based Importance Scoring

## Problem

**TriAttention** (`WeianMao/triattention`; the README badge cites arXiv:2604.04921) scores KV tokens by **phase coherence** rather than by raw attention weights. It first **inverts RoPE** on each key back to its pre-rotation complex form, then multiplies it by the mean query in the complex plane:

\[
\tilde k = k_{\text{post}}\cdot e^{-i\theta},\qquad
r = q_{\text{mean}}\cdot \overline{\tilde k}.
\]

The magnitude of \(r\) measures how well the query and key align, while the phase \(\varphi=\arg r\) measures their direction mismatch. The score multiplies the magnitude by \(\cos(\Delta t\cdot\omega+\varphi)\): the first few frequency bins also receive a temporal phase \(\Delta t\cdot\omega\), where \(\Delta t\) is built from geometric offsets \(\{1,2,4,\dots\}\); the remaining bins see only the spatial phase. Finally, scores are z-normalized per `(layer, head)` along tokens, averaged across layers, and top-k selected per head.

## Rules

Implement `solve(q_mean, q_abs, k, rope_cos, rope_sin, omega, pos, keep, current_end_frame, frame_seq_length=1, temporal_count=None, offset_max_frames=128, score_aggregation="mean", normalize_scores=True)`:

### Inputs

- `q_mean`: \((L,H,F,2)\), the mean query complex per `(layer, head, freq)`; the last axis is `(real, imag)`.
- `q_abs`: \((L,H,F)\), the query magnitude baseline at the same positions; it supplies an additive term independent of phase.
- `k`: \((L,T,H,2F)\), keys **after RoPE**; the last axis packs `(real, imag)` adjacently, so frequency \(f\) uses indices \(2f,2f+1\).
- `rope_cos`, `rope_sin`: \((T,F)\), \(\cos\theta\) and \(\sin\theta\) per token and frequency.
- `omega`: \((F,)\), angular velocity per frequency.
- `pos`: a \((T,)\) integer tensor (must not be floating point).
- `keep`: `int`, number of tokens kept per head, \(0\le \text{keep}\le T\).
- `current_end_frame`: `int`, the end frame of the current window.
- `frame_seq_length`: `int >= 1`, number of tokens per frame, default 1.
- `temporal_count`: `int`, how many leading frequency bins receive the temporal phase; when `None`,
  \[
  F-2\lfloor F/3\rfloor .
  \]
- `offset_max_frames`: `int >= 0`, upper bound of the geometric offsets, default 128.
- `score_aggregation`: `"mean"` (default) or `"max"`, aggregation over offsets.
- `normalize_scores`: `bool`, whether to z-normalize per `(layer, head)`, default `True`.

All floating inputs must share one dtype (float32 / float64); `L/H/F/T` must agree; `keep`, frame length, and offset bounds must be `int` (`bool` is rejected). Invalid shapes, dtypes, or ranges raise `ValueError`.

### Computation

1. **Invert RoPE**: reshape `k` to \((L,T,H,F,2)\) and, for each frequency,
   \[
   \tilde k_{\text{re}}=k_{\text{re}}\cos\theta+k_{\text{im}}\sin\theta,\qquad
   \tilde k_{\text{im}}=k_{\text{im}}\cos\theta-k_{\text{re}}\sin\theta .
   \]
2. **Phase and amplitude**:
   \[
   \varphi=\operatorname{atan2}\!\left(\Im(q_{\text{mean}}\overline{\tilde k}),\,
   \Re(q_{\text{mean}}\overline{\tilde k})\right),
   \]
   \[
   A=|q_{\text{mean}}|\,|\tilde k|,\qquad
   E=(q_{\text{abs}}-|q_{\text{mean}}|)\,|\tilde k| .
   \]
3. **Geometric offsets**: take all \(\{1,2,4,\dots\}\) that are \(\le\) `offset_max_frames`; if none exist (`offset_max_frames=0`), fall back to \(\{1\}\). With token \(t\)'s frame \(\lfloor pos_t/\text{frame\_seq\_length}\rfloor\),
   \[
   \Delta t_{t,o}=\text{current\_end\_frame}-\left\lfloor \frac{pos_t}{\text{frame\_seq\_length}}\right\rfloor+o .
   \]
   `delta_t` has shape \((T,O)\), where \(O\) is the number of offsets.
4. **Temporal mask**: the first `temporal_count` frequency bins get 1, the rest 0; call it \(m_f\).
5. **Per-offset score**:
   \[
   b_{l,h,t,o}=\sum_f A_{l,h,t,f}\cos\!\big(\Delta t_{t,o}\,m_f\,\omega_f+\varphi_{l,h,t,f}\big),
   \]
   \[
   \text{score}_{l,h,t}=\operatorname*{mean}_o\Big(b_{l,h,t,o}+\sum_f E_{l,h,t,f}\Big).
   \]
   With `score_aggregation="max"`, replace \(\operatorname*{mean}_o\) by \(\max_o\); the additive term \(\sum_f E\) does not depend on the offset.
6. **Normalization**: when `normalize_scores=True`,
   \[
   \text{normalized}_{l,h,t}=\frac{\text{score}_{l,h,t}-\mu_{l,h}}
   {\max(\sigma_{l,h},\,10^{-6})},
   \]
   where \(\mu,\sigma\) are computed along tokens and \(\sigma\) is the **population** standard deviation (`unbiased=False`).
7. **Layer averaging**: `head_score = normalized.mean(dim=0)` of shape \((H,T)\).
8. **Per-head top-k**: take the top `keep` entries of `head_score` in descending order (ties go to the smaller token index), then sort the selected indices ascending.

### Returns

A 6-key dict:

- `phi`: \((L,H,T,F)\), the relative phase \(\varphi\);
- `delta_t`: \((T,O)\), the frame difference after geometric offsets;
- `score`: \((L,H,T)\), the aggregated raw score;
- `normalized`: \((L,H,T)\), the z-normalized score (equal to `score` when normalization is off);
- `head_score`: \((H,T)\), the per-head score after layer averaging;
- `kept`: \((H,\text{keep})\) int64, each row ascending and duplicate-free.

Everything except `kept` keeps the input dtype. Internal arithmetic is float64 and no input may be modified in place. `keep=0` returns `kept` of shape \((H,0)\).

## Example

```
L=1, H=1, F=2, T=2
q_mean = [[[[1, 0], [0, 0]]]]        # (1,1,2,2)
q_abs  = [[[1, 0]]]                  # (1,1,2)
k      = [[[[1, 0, 1, 0]],           # token 0: freq0=(1,0), freq1=(1,0)
           [[0, 1, 1, 0]]]]          # token 1: freq0=(0,1), freq1=(1,0)
rope_cos = ones(2, 2), rope_sin = zeros(2, 2)
omega = [1.0, 0.0], pos = [0, 1]
keep = 1, current_end_frame = 2, temporal_count = 1, offset_max_frames = 2
(other defaults: frame_seq_length=1, score_aggregation="mean", normalize_scores=True)

phi        = [[[[0.0, 0.0], [-1.5707963267948966, 0.0]]]]     # (1,1,2,2)
delta_t    = [[3.0, 4.0], [2.0, 3.0]]                          # (2,2)
score      = [[[-0.8218180587320287, 0.5252087174427744]]]     # (1,1,2)
normalized = [[[-1.0, 1.0]]]                                   # (1,1,2)
head_score = [[-1.0, 1.0]]                                     # (1,2)
kept       = [[1]]                                             # (1,1) int64
```

`offset_max_frames=2` yields offsets \(\{1,2\}\): token 0 has \(\Delta t=2+\{1,2\}=\{3,4\}\), token 1 has \(1+\{1,2\}=\{2,3\}\). `temporal_count=1` gives frequency 0 the temporal phase and freezes frequency 1. Token 0 has \(\varphi=0\) and amplitude 1, so its score is \((\cos 3+\cos 4)/2=-0.821818\ldots\); token 1 has \(\varphi=-\pi/2\), so its score is \((\cos(2-\pi/2)+\cos(3-\pi/2))/2=0.525208\ldots\). After z-normalizing two tokens the values are exactly \([-1,1]\), and `keep=1` retains the higher-scoring token 1.

## Constraints

- CPU float32 / float64; all floating inputs share one dtype; the last axis of `q_mean` must be 2 and that of `k` must be \(2F\).
- `pos` must be a 1D integer tensor; `keep`, `current_end_frame`, `frame_seq_length`, and `offset_max_frames` must be `int`; `normalize_scores` must be `bool`.
- `0 <= keep <= T`; `frame_seq_length >= 1`; `offset_max_frames >= 0`; `0 <= temporal_count <= F`; `score_aggregation` is `"mean"` or `"max"`.
- The offset set is \(\{1,2,4,\dots\}\) capped by `offset_max_frames`; when empty, use \(\{1\}\).
- The z-normalization uses the population standard deviation clamped to \(10^{-6}\): a constant row (including \(T=1\)) normalizes to 0 and must not produce NaN / Inf.
- Ties always go to the smaller token index; every `kept` row is ascending and duplicate-free.
- Invalid shapes, dtypes, non-integer `pos`, out-of-range `keep` / `temporal_count`, or an unsupported aggregation raise `ValueError`.
