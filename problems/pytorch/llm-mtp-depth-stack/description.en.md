# MTP: Multi-Token Prediction Depth Stack

## Problem Statement

A standard language model predicts only the next token at each step. **Multi-Token Prediction** (MTP) hangs \(D\) extra lightweight modules on top of the backbone so that a single backbone hidden state predicts the next \(D+1\) tokens at once — denser training signal, and more pressure to "plan before writing".

DeepSeek-V3's MTP is a **serial depth stack**: module \(k\) does not consume the backbone state \(h^0\); it consumes the previous module's state \(h^{k-1}\), fused with the input embedding of the token \(k\) steps ahead:

\[ h^{k}=M_k\big[\ \mathrm{RMS}(h^{k-1})\ ;\ \mathrm{RMS}(\mathrm{Emb}(x_{i+k}))\ \big] \]

Both halves are normalized separately, concatenated (width \(2d\)), and projected back to \(d\). \(M_k\) is followed by a shared output head and embedding; this problem covers only the \(M_k\) stage.

Two details are easy to miss: the second input is the **token embedding, not the model's own output** (it shares the backbone's embedding table), and **each depth reads the embedding shifted by one more position** — depth \(k\) uses \(x_{i+k}\), not \(x_{i+k-1}\).

## Rules

Inputs:

- `hidden`: \((B,S,d)\), the backbone's last hidden state, denoted \(h^0\);
- `embeddings`: \((B,L,d)\), the input embedding sequence; it must reach \(D\) tokens ahead, so \(L\ge S+D\);
- `W`: \((D,2d,d)\), the per-depth fusion projection applied to the \(2d\)-dimensional concatenation \([\,\mathrm{RMS}(h^{k-1});\mathrm{RMS}(\text{embeddings})\,]\);
- `eps`: the RMSNorm stabilizer, a finite positive real.

Recursion (\(k=1,\dots,D\)):

\[ \mathrm{RMS}(x)=\frac{x}{\sqrt{\frac1d\sum_j x_j^2+\varepsilon}}\ ,\qquad h^{k}_{b,s}=W_k^{\top}\Big[\mathrm{RMS}\big(h^{k-1}_{b,s}\big);\ \mathrm{RMS}\big(\text{embeddings}_{b,\,k+s}\big)\Big] \]

That is, depth \(k\) reads `embeddings[:, k : k+S]` (left-closed, right-open). RMS is taken along the feature dimension and carries **no learnable scale**.

Return a tuple of length \(D\) holding \(h^1,\dots,h^D\) in order, each a float32 tensor of shape \((B,S,d)\).

## Example

\[ h^0=\begin{bmatrix}1.0&2.0\\0.5&-1.0\end{bmatrix},\quad \mathrm{Emb}=\begin{bmatrix}0.5&0.5\\1.0&-1.0\\0.25&0.75\\-1.0&0.5\end{bmatrix} \]

(\(B=1\), \(S=2\), \(D=2\), \(d=2\); the embedding has 4 rows.) With

\[ W_1=\begin{bmatrix}1&0\\0&1\\0.5&0\\0&0.5\end{bmatrix},\qquad W_2=\begin{bmatrix}0&1\\1&0\\0.5&0.5\\1&0\end{bmatrix} \]

and `eps=1e-6`:

```
h^1 -> [[[1.132455, 0.764911], [0.856061, -0.594091]]]
h^2 -> [[[2.356817, 1.395533], [-0.806298, 0.529388]]]
```

## Constraints

- All three inputs must be tensors with fixed rank: `hidden` / `embeddings` / `W` are 3-D, otherwise raise `ValueError`.
- Shapes must be consistent: `embeddings` shares the batch and feature dim with `hidden`; `W` is \((D,2d,d)\) with \(D\ge1\); `embeddings` has length at least \(S+D\).
- `eps` must be a finite positive real; `bool` is rejected.
- **Structural properties**: RMSNorm is invariant to positive rescaling — multiplying `hidden` by any positive constant leaves \(h^1\) unchanged, and so does rescaling the embedding slice that depth \(k\) reads. Depth \(k\) reads `embeddings[:, k:k+S]`: the embedding at position 0 is never read again (that token was already consumed into \(h^0\)), position \(S+1\) lands only in depth 2's window, and position 1 lands in depth 1's window. Changing depth 2's weights leaves \(h^1\) untouched but changes \(h^2\).
- Compute in float64 internally if you like, but return float32.
