# Mechanism: Exponential Moving Average (EMA) Weights with Warmup

## Problem

Many training recipes (diffusion models, Mean Teacher self-training, Model Soup, evaluation-time weight averaging) keep an extra set of **Exponential Moving Average (EMA)** weights: they are not part of the autodiff graph, each step they just absorb the "real training weights" very slowly, producing a smoother weight trajectory with better generalization that is usually what gets evaluated or used for generation. So that the EMA is not "locked in" by the badly initialized weights in the first few steps, a **warmup** is commonly added: early on a smaller decay is used (following the new weights faster), and the decay then ramps up to its configured value.

This problem asks you to implement an EMA shadow-weight maintainer: it holds shadow copies in one-to-one correspondence with the training parameters, and every call to `update()` smooths the current parameters into the shadows in place. **Note**: EMA does not update the training parameters and does not use gradients; it is a mechanism object independent of `Optimizer`.

## Rules

- Factory function `solve(params, decay=0.9, warmup=True)`, returning an EMA object that must expose:
  - `.shadow`: a **list** in one-to-one correspondence with `params`; at construction each shadow tensor is initialized as a **copy of the current value of its parameter** (`shadow_i \leftarrow p_i`).
  - `.update()`: performs one moving-average step (reads the **current** `params` values and updates `.shadow` in place).
- Number the `update()` calls by \(n\) (\(n=1,2,3,\dots\)); the effective decay of this step is:
  \[d_n=\begin{cases}\min\!\left(decay,\ \dfrac{1+n}{10+n}\right), & \texttt{warmup}\\[6pt] decay, & \text{\texttt{warmup}=False}\end{cases}\]
  then for each parameter: \[shadow_i \leftarrow d_n\cdot shadow_i + (1-d_n)\cdot p_i\quad(\text{in place})\]
- `update()` **does not modify** `params` and does not zero anything; it only advances the internal counter \(n\) and updates `.shadow`. Shadow tensors are **separate** from the training parameters (they are clones, not the same memory).
- Everything is float32; using an off-the-shelf EMA library is forbidden.

## Example

A single parameter \(p=[1.0,\,-2.0,\,0.5]\), `decay=0.9`, `warmup=True`. Simulate training: each step adds \([0.1,\,-0.05,\,0.2]\) to \(p\) and then calls `update()` once, for 7 calls in total. Warmup keeps early \(d_n\) small (\(d_1=2/11\approx0.18\)) so the shadows follow quickly, and \(d_n\) approaches its cap as \(n\) grows but stays below 0.9 within the 7 steps:

```
shadow after 7 update calls ≈ [1.620004, -2.310002, 1.740008]
```

## Constraints

- \(n\) starts at \(1\) (at construction \(n=0\) and \(shadow=p_0\)).
- Must agree element by element with the reference implementation (see notes.md for the comparison setup).
- Supports several parameters of different shapes; warmup shares the same \(d_n\) across all parameters.
