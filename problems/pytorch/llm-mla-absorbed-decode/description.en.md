# MLA Decode: Latent-Space Attention via Weight Absorption

## Problem Statement

DeepSeek's **Multi-head Latent Attention** (MLA) compresses the KV state into a low-dimensional latent vector \(c_{KV}\in\mathbb R^{d_c}\) kept in the cache, and two per-head projections lift it back into keys and values:

\[ k^{\text{nope}}_h=W_{UK,h}\,c_{KV},\qquad v_h=W_{UV,h}\,c_{KV} \]

Note that \(W_{UK}\) only touches the **non-RoPE** key component; positional information comes from a separate \(d_r\)-dimensional RoPE key \(k^{PE}\) shared by all heads.

The naive decode materializes \(k^{\text{nope}}\) and \(v\) (of size \(d_h\) per head) for every cached token on every step. **Weight absorption** instead pushes the lifting to both ends of the attention:

\[ \underbrace{q_h^\top W_{UK,h}}_{\text{folded into the query}}\;c_{KV}\;+\;q^{PE}_h\cdot k^{PE} \]

Since \(W_{UK}\) does not depend on the token, it can be multiplied into the query first, giving \(\tilde q_h=q_h^\top W_{UK,h}\in\mathbb R^{d_c}\); scores are then computed directly against \(c_{KV}\) in the \(d_c\)-dimensional latent space. Likewise the weighted sum of values can be done in latent space (\(\sum_t p_t c_{KV,t}\)) with a **single** \(W_{UV,h}\) lift per head at the end. Decode then never materializes an \(S\times H\times d_h\) key/value tensor.

## Rules

Inputs:

- `q_nope`: \((B,H,d_h)\), the per-head non-RoPE query (already through \(W_{UQ}\));
- `q_pe`: \((B,H,d_r)\), the per-head RoPE-applied query;
- `c_kv`: \((B,S,d_c)\), the latent KV cache;
- `k_pe`: \((B,S,d_r)\), the RoPE key shared by all heads;
- `W_UK`, `W_UV`: both \((H,d_h,d_c)\) — **row \(k\)** of \(W_{UK,h}\) holds the coefficients of the \(k\)-th coordinate of \(k^{\text{nope}}\) over \(c_{KV}\);
- `scale`: the pre-softmax scale, a finite positive real.

Computation:

1. \(\tilde q_h=q_h^\top W_{UK,h}\), i.e. \(\tilde q_h[c]=\sum_k q_h[k]\,W_{UK,h}[k,c]\), giving \((B,H,d_c)\);
2. scores \(s_{h,t}=\big(\tilde q_h\cdot c_{KV,t}+q^{PE}_h\cdot k^{PE}_t\big)\cdot\text{scale}\), softmaxed into \(p_{h,t}\);
3. latent-space weighted sum \(z_h=\sum_t p_{h,t}\,c_{KV,t}\), giving \((B,H,d_c)\);
4. one lift per head, \(o_h=W_{UV,h}\,z_h\).

Return a float32 tensor of shape \((B,H,d_h)\).

## Example

\[ q^{\text{nope}}=\begin{bmatrix}1.0&0.5\\0.0&-1.0\end{bmatrix},\quad q^{PE}=\begin{bmatrix}0.5&0.5\\-0.5&0.5\end{bmatrix},\quad c_{KV}=\begin{bmatrix}1.0&0.0\\0.5&1.0\\-1.0&0.5\end{bmatrix},\quad k^{PE}=\begin{bmatrix}0.0&1.0\\1.0&0.0\\0.5&0.5\end{bmatrix} \]

\[ W_{UK,0}=\begin{bmatrix}1.0&0.0\\0.0&1.0\end{bmatrix},\ W_{UK,1}=\begin{bmatrix}0.5&0.5\\0.0&1.0\end{bmatrix},\quad W_{UV,0}=\begin{bmatrix}1.0&0.5\\0.0&1.0\end{bmatrix},\ W_{UV,1}=\begin{bmatrix}0.5&0.0\\1.0&1.0\end{bmatrix} \]

(with \(H=2\), \(d_h=d_r=d_c=2\), \(S=3\), \(B=1\); the subscript on each matrix is the head index). With `scale=0.5`:

```
returned shape (1, 2, 2):
[[[0.698159, 0.500000],
  [0.146223, 0.632368]]]
```

## Constraints

- All six inputs must be tensors with fixed rank: `q_nope`/`q_pe`/`c_kv`/`k_pe`/`W_UK`/`W_UV` are all 3-D, otherwise raise `ValueError`.
- Shapes must be consistent: `q_pe` shares the batch and head count with `q_nope`; `k_pe` is \((B,S,d_r)\) matching `c_kv`'s batch and sequence length; `W_UK` and `W_UV` are both \((H,d_h,d_c)\).
- `scale` must be a finite positive real; `bool` and strings are rejected.
- **Equivalence**: the result must match, value for value, the path that lifts \(k^{\text{nope}},v\) per token and then runs ordinary attention (to float32 precision). With \(S=1\) the softmax is identically 1, so the output is exactly \(W_{UV,h}\,c_{KV,0}\), independent of the query; with zero queries and zero RoPE keys the weights are uniform, so the output is \(W_{UV,h}\) applied to the mean of \(c_{KV}\).
- Heads are independent: changing one head's \(W_{UK}\) / \(W_{UV}\) must not affect the other heads' outputs.
- Compute in float64 internally if you like, but return float32.
