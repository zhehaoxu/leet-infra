# RoPE (Rotary Position Embedding) Explained

## Core Idea

Instead of adding a position vector to each token, treat each pair of adjacent dimensions as a vector in a 2-D plane and rotate it by an angle determined by its **position** $m$. Rotation is an orthogonal transform that preserves magnitude, and when two vectors are dotted the result depends only on the **relative angle difference** — so **relative position** is encoded naturally.

$$ \text{RoPE}(x,m)=\big(x_{2i}\cos m\theta_i - x_{2i+1}\sin m\theta_i,\ \ x_{2i}\sin m\theta_i + x_{2i+1}\cos m\theta_i\big)_{i} $$

$$ \theta_i = \text{base}^{-2i/d},\qquad i=0,\dots,\tfrac d2-1 $$

## Why It Encodes "Relative Position"

Within the same head, dot the rotated $q$ at position $m$ with $k$ at position $n$; properties of rotation matrices give $R_m q\cdot R_n k = q^\top R_{n-m} k$ — the result involves only $n-m$. This is the key reason RoPE makes attention sensitive to "distance" but insensitive to "absolute position", and why it can extrapolate / work with NTK-aware and YaRN scaling.

## Implementation Pitfalls (common mistakes in this problem)

1. **Pairing scheme**: this implementation pairs dimensions **two by two, adjacent** (even dims `x[...,0::2]` as $x_1$, odd dims `x[...,1::2]` as $x_2$). Note there is also the "front-half/back-half pairing" (the common `half` split in GPT-NeoX/LLaMA); the two conventions require different weight arrangements — do not mix them.
2. **Angular frequency**: $\theta_i=\text{base}^{-2i/d}$; the exponent is $-2i/d$, do not write $-i/d$ (that is the mistaken per-half-dimension version missing the factor 2); low-frequency dimensions change slowly, high-frequency dimensions change fast.
3. **Broadcast positions over batch/heads**: `positions.view(-1,1)*theta.view(1,-1)` yields $(L,\text{half})$, then multiply with $x$'s $(...,L,\text{half})$ via broadcasting.
4. When concatenating back, use `stack(...,dim=-1).flatten(-2,-1)` to guarantee the interleaved dimension order matches the split.

## Applications

Mainstream LLMs such as LLaMA, GPT-NeoX, and PaLM all use RoPE. Once you understand "per-pair rotation + relativity", the contrast with ALiBi (purely additive distance bias, see the "ALiBi" problem) is stark: one modifies Q/K, the other modifies scores.
