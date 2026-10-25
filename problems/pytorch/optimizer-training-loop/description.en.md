# SGD Training Loop and Numerical Reproduction

## Problem

Every line of a training loop determines the numerical result: when gradients are cleared, when the loss is recorded, how the `momentum` buffer accumulates. Miss one `zero_grad()` or record at a slightly different point, and the loss trajectory is completely different. This problem asks you to reproduce the loss trajectory of a standard training loop exactly.

## Rules

Implement `solve() -> list[float]`, execute strictly according to the following flow, and return the loss sequence:

1. Parameter: $w = \texttt{torch.tensor([2.0], requires\_grad=True)}$
2. Optimizer: `torch.optim.SGD([w], lr=0.1, momentum=0.9)`
3. Loop for 20 steps, with each step in order:
   - `opt.zero_grad()`
   - Compute the objective $loss = (w^2).\mathrm{sum}()$
   - **Record the current loss value** (`loss.item()`, before `backward()`)
   - `loss.backward()`
   - `opt.step()`
4. Return a `list[float]` of length 20.

Notes:

- Recording happens before `backward`/`step`, so item 0 is the loss of the initial $w=2.0$, which is $=4.0$.
- **Do not** touch the global seed; this flow has no randomness, and the result is a deterministic float sequence.
- Momentum makes $w$ overshoot the minimum $0$ and oscillate back and forth — the rebound in items 4~7 is a direct manifestation of momentum overshoot.

## Examples

```
Input: none
Output (first 5 items): [4.0, 2.56, 0.8464, 0.0154, 0.3809, ...]
```

Deriving step 0→1: $loss_0 = 2.0^2 = 4.0$; $\nabla w = 2w = 4.0$; momentum buffer $b = 0.9 \cdot 0 + 4.0 = 4.0$; $w \leftarrow 2.0 - 0.1 \cdot 4.0 = 1.6$; $loss_1 = 1.6^2 = 2.56$.

## Constraints

- Return `list[float]` (not a tensor) of exactly length 20
- The grader compares the full sequence element-wise with `atol=1e-3`
