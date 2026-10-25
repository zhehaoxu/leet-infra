# ALiBi (Attention with Linear Biases) Explained

## Core idea

Rather than injecting position vectors or using rotations, it adds a **distance-proportional** penalty directly to the attention scores, making "the farther away, the less attention" an inductive bias:

$$ A_{ij}^{(h)} = q_i^{(h)}\cdot k_j^{(h)} - m_h\,|i-j| $$

$m_h$ is the **slope** of head $h$; each head has a different slope → a different distance-decay scale. The additive bias sits before softmax, which is equivalent to exponentially down-weighting distant positions.

## Decreasing slopes (the big trap of this problem — do not reverse the direction)

The paper uses a geometric sequence for $n$ heads:

$$ m_h = 2^{-8h/n},\quad h=1,\dots,n $$

i.e. it **decreases** from $2^{-8/n}$ to $2^{-8}$ — the larger the head index, the smaller the slope, the flatter the decay, and the farther that head sees. So the assertion should be "the slope is monotonically **decreasing** with head index"; writing it the other way (as increasing) is a classic mistake.

## Distance matrix

$\text{dist}_{ij}=|i-j|$: use `arange` for rows and columns, subtract and take the absolute value to get $(L,L)$; then multiply by the per-head slopes `slopes.view(-1,1,1)` and broadcast to $(H,L,L)$. This implementation returns exactly this bias matrix (including the minus sign `-slope*dist`), which can be added directly to the scores.

## Why it deserves its own problem

- **Zero parameters**, purely additive: trained on short sequences, it stays stable when extrapolating to longer sequences at inference (better extrapolation than sinusoidal encodings/learnable position embeddings), and has been used in BLOOM and MPT.
- It is a different route from RoPE: ALiBi only modifies the scores and leaves Q/K untouched; RoPE modifies Q/K so that the dot product itself carries relative position.

## Pitfalls

`dtype` and `device` follow the input; in causal/lower-triangular settings the distance for $i<j$ is already masked out, so the bias only acts on the allowed lower half — the two stack independently.
