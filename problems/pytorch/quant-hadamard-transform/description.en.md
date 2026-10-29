# Quantization: Fast Recursive Hadamard Transform Construction

## Description

The **Hadamard transform** (Walsh-Hadamard) is used in QuaRot and other quantization/rotation methods to rotate activations and weights into an isotropic direction. The Sylvester matrix of size \(2^k\) is defined recursively as

\[H_1=[1],\qquad H_{2n}=\begin{bmatrix}H_n & H_n\\ H_n & -H_n\end{bmatrix}.\]

After normalization, \(H_n/\sqrt n\) is orthogonal: \((H_n/\sqrt n)^{\mathsf T}(H_n/\sqrt n)=I\).

When the dimension \(n\) is not a power of two, use a **block-diagonal** approximation: greedily decompose \(n\) into a sum of powers of two, place a normalized Hadamard block of the corresponding size on each diagonal block, and apply a random sign diagonal matrix \(D=\operatorname{diag}(s_1,\dots,s_n)\) with \(s_i\in\{-1,+1\}\). The final transform is

\[y=H^{(n)} D x.\]

Random signs are generated from `seed` via `torch.Generator` for reproducibility.

## Task

Implement `solve(n, x, seed=0)`:

- `n`: positive integer dimension, matching `x.shape[-1]`.
- `x`: float32 with shape \((n,)\) or \((b, n)\).
- `seed`: non-negative integer for sign generation.
- Return `(H, y)`:
  - `H`: float32 with shape \((n, n)\), the normalized Hadamard / block-diagonal Hadamard matrix;
  - `y`: float32 with the same shape as `x`, equal to \(H D x\).

## Hints

- Recursively build \(H_{2n}\) with `torch.cat` of `[H, H]` and `[H, -H]`, then divide the whole matrix by \(\sqrt n\).
- For non-power-of-two dimensions, decompose \(n=2^{k_1}+2^{k_2}+\dots\) greedily from largest to smallest and use `torch.block_diag`.
- Generate signs with `torch.randint(0, 2, (n,), generator=g) * 2 - 1`.
