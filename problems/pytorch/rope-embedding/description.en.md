# RoPE (Rotary Position Embedding)

## Problem Description

RoPE injects positional information by **rotating each adjacent 2-D subspace** into which the vector is split by position, each through its own angle, so that the attention score between two tokens depends only on their **relative position**.

For the last dimension $d$, the $i$-th component pair $(x_{2i}, x_{2i+1})$ undergoes a 2D rotation by angle $p\cdot\theta_i$ at position $p$, where the rotation frequency is

$$ \theta_i = \text{base}^{-2i/d} $$

and the corresponding rotation is

$$ \begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} =
\begin{pmatrix} \cos(p\theta_i) & -\sin(p\theta_i) \\ \sin(p\theta_i) & \cos(p\theta_i) \end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix} $$

## Rules

- Implement `solve(x, positions, base=10000.0)`; `x` has shape `(..., L, d)`, `positions` has shape `(L,)`, return the same shape.
- **Adjacent pairing** (the interleaved convention of rotate-half): $(0,1),(2,3),\dots$, i.e. each even index pairs with its successor, each using angle $p\cdot\text{base}^{-2i/d}$ ($i$ is the pair index, starting from 0).
- The last dimension $d$ is even; the frequency exponent of pair $i$ is $-2i/d$ (equivalent to normalizing `base ** (-(0::2)/d*2)`).
- At position $p=0$ the transform is the identity ($\cos 0=1,\sin 0=0$).

## Example

```
Input: x=[[[1,0]]], positions=[1], base=10000, d=2
   i=0 → θ=10000^0=1 → angle=1*1=1rad → rotate (1,0) → (cos1, sin1)
Output: [[[0.5403023, 0.8414710]]]

Input: x=[[[1,0,0,1]]], positions=[2], d=4
   pair0 θ=10000^0=1, angle=2 → (1,0)→(cos2, sin2)=(-0.4161, 0.9093)
   pair1 θ=10000^(-0.5)=0.01, angle=0.02 → (0,1)→(-sin0.02, cos0.02)=(-0.0200, 0.9998)
Output: [[[-0.4161468, 0.9092974, -0.0199987, 0.9998000]]]
```

## Constraints

- float32/float64 input; $d$ is even
- Consistent with the complex-number implementation `view_as_complex(x) * exp(1j * p * θ)`
