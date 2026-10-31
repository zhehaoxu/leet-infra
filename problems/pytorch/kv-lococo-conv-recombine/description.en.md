# KV Cache: LoCoCo Convolutional Recomputation Merging

## Problem

H2O-style methods score tokens by accumulated attention and cut everything below a threshold. But a low score does not mean the token is useless: a dropped token can often be approximated by a linear combination of its neighbours, and deleting it outright is a permanent loss. LoCoCo (2406.05317) therefore compresses in two stages, **hard keep plus soft recompute**:

1. keep the heavy hitters under the H2O rule, and always keep the trailing `local_len` tokens;
2. among the remaining non-heavy-hitter tokens, pick the `mem_compress` highest-scoring ones as residual representatives, and feed all the rest into a 1-D convolution sliding along the **token axis** that learns soft weights and squeezes them into the same number of summary tokens;
3. blend the convolutional weights with the one-hot choice of representatives through `normalizer` to get the final weights, then multiply them back into K/V.

Because the kernel slides along the token axis, every summary token sees all non-heavy-hitter K/V in a neighbourhood; `[K;V]` are concatenated on the same channel axis, so one convolution can freely mix key and value channels. Upstream the kernel and `normalizer` are trained parameters; this problem takes them as **explicit inputs** and trains nothing.

## Task

Implement `solve(keys, values, scores, conv_weight, normalizer, hh_keep_rate, local_len, mem_size)`:

- `keys`/`values`: `(S,H,D)`, same dtype (float32/float64), NHD layout;
- `scores`: `(H,S)` accumulated attention scores, same dtype, larger is more important;
- `conv_weight`: `(m,2D,K)` with odd positive `K`;
- `normalizer`: a scalar (float or 0-dim tensor);
- `hh_keep_rate ∈ (0,1)`, `local_len ≥ 0`, `mem_size ≥ 2`, all integers.

Let \(m=\lfloor \texttt{mem\_size}\,(1-\texttt{hh\_keep\_rate})\rfloor\) (truncation toward zero), \(\textit{keep}=\texttt{mem\_size}-m\), \(n=S-\textit{keep}\). The heavy-hitter set is

\[
\mathcal H=\operatorname*{TopK}_{\textit{keep}-\texttt{local\_len}}\{s_i:0\le i<S-\texttt{local\_len}\}\ \cup\ \{S-\texttt{local\_len},\dots,S-1\}
\]
picked by descending score with ties going to the smaller token index, then returned **in ascending token order**. The residual set \(\mathcal R\) is the \(m\) highest-scoring members of the complement \(\bar{\mathcal H}\), also ascending. The convolution is

\[
c_{o,t}=\sum_{d=0}^{2D-1}\sum_{j=0}^{K-1}w_{o,d,j}\,x_{d,\,t+j-p},\qquad p=\frac{K-1}{2},\qquad x=[k_{\bar{\mathcal H}};v_{\bar{\mathcal H}}]
\]
its output length stays \(n\), with \(p\) zeros on each side. The weights are

\[
q_{o,t}=\frac{\exp(\max(c_{o,t},0))}{\sum_{t'}\exp(\max(c_{o,t'},0))},\qquad
b_{r,t}=(1-\nu)\,\mathbf 1[t=\pi(r)]+\nu\,q_{r,t}
\]
where \(\pi(r)\) is the position of the \(r\)-th residual representative inside the ascending \(\bar{\mathcal H}\). Finally

\[
K'=[k_{\mathcal H};\,b\,k_{\bar{\mathcal H}}],\qquad V'=[v_{\mathcal H};\,b\,v_{\bar{\mathcal H}}]
\]
Return a 6-key dict: `hh_idx`, `res_idx` (int64), `conv_prob`, `blend`, `merged_keys`, `merged_values`. Compute in float64 and return floats in the input dtype; never mutate the inputs.

## Example

```
keys = [[[1, 0]], [[0, 3]], [[1, 3]], [[-1, 3]], [[0, 3]], [[2, 2]]]   # S=6, H=1, D=2
values = [[[1, 2]], [[2, 5]], [[0, 5]], [[1, 5]], [[-1, 5]], [[.5, .5]]]
scores = [[.9, .1, .7, .2, .6, 5.]]      # the last entry is local: kept, never competing
mem_size = 4, hh_keep_rate = .5          # m = 2, keep = 2, n = 4
local_len = 1; normalizer = .5; K = 3     # the prefix only elects keep-local = 1 heavy hitter
conv_weight[0, 1, 1] = 1, conv_weight[1, 3, 1] = 1   # only the centre tap of k1 / v1

hh_idx   = [[0, 5]]          # token 0 wins the prefix; token 5 is always kept
res_idx  = [[2, 4]]          # the two best-scoring members of the complement (.7 and .6)
conv_prob = [[[.25, .25, .25, .25],      # the centre tap reads the constant channels 3 / 5
              [.25, .25, .25, .25]]]     # so softmax is uniform
blend    = [[[.125, .625, .125, .125],   # 0.5*one-hot(res) + 0.5*conv_prob
             [.125, .125, .125, .625]]]
merged_keys   = [[[1, 0], [2, 2], [.5, 3], [0, 3]]]
merged_values = [[[1, 2], [.5, .5], [.25, 5], [-.25, 5]]]
```

## Constraints

- **Padding and output length**: same padding in `nn.Conv1d` means `(K-1)//2` zeros on each side — 10 per side for kernel 21; only odd positive kernels are accepted, otherwise the output is one shorter. It is cross-correlation: position `t` reads `x[t+j-p]`, the kernel is not flipped.
- **`int()` truncates toward zero, it does not round**: the fractional part of `mem_size*(1-rate)` is discarded, so `mem_size=10, rate=0.75` gives `m=2`. `keep` is recovered as `mem_size-m`, and the two always sum to `mem_size`.
- **Too few residuals is an error**: when fewer than `m` non-heavy-hitter tokens remain, upstream `topk` crashes; this problem raises `ValueError`.
- **Ties go to the smaller token index**: both topk calls order by descending score with ties on the smaller token index; `torch.topk` tie order must not be relied on.
- **Heavy hitters are in token order**: upstream collects them with `masked_select`, which is ascending token order rather than score order; the residual one-hot rows are likewise ordered by ascending representative token.
- **Local tail does not compete**: the prefix topk has `keep-local_len` slots and the tail is set to True directly, so the tail can never steal a prefix slot.
- **Softmax runs on the token axis after ReLU**: each output channel normalizes over all non-heavy-hitter tokens, so every `blend` row sums to 1; `normalizer=0` degenerates to pure selection and `normalizer=1` to pure convolution; upstream defaults to `hidden_act='relu'`, applying `max(c,0)` before softmax, otherwise negative weights would break convexity.
- **`conv_weight` is a full-channel convolution**: `(m,2D,K)` mixes all 2D K/V channels at once, not depthwise; the bias of the upstream `Conv1d` is treated as zero here.
