# Assembling an MLP Block

## Problem Description

`nn.Module` is PyTorch's basic unit for organizing models: you declare submodules and parameters in `__init__` and describe the data flow in `forward`. The framework collects parameters and tracks gradients automatically, so `model.parameters()` can be handed straight to an optimizer.

## Rules

Implement `solve(in_features: int, hidden_multiplier: int, out_features: int) -> torch.nn.Module`, returning a **two-layer MLP block** module instance:

$$\text{Linear}(in \to in \cdot m) \;\rightarrow\; \text{ReLU} \;\rightarrow\; \text{Linear}(in \cdot m \to out)$$

Requirements:

1. Implement it as an `nn.Module` subclass (or assemble it with `nn.Sequential`), using only `nn.Linear` and `nn.ReLU` inside.
2. The hidden layer width is `in_features * hidden_multiplier`.
3. Raise `ValueError` when `hidden_multiplier < 1`.
4. The returned module must already have its parameters initialized (so it can be called for a forward pass directly).

## Example

```
Input: in_features=8, hidden_multiplier=4, out_features=2
Module: Linear(8→32) + ReLU + Linear(32→2)
Trainable parameter count: 8*32+32 + 32*2+2 = 354
```

Parameter count check: first layer weights $8 \times 32$ + bias $32$, second layer weights $32 \times 2$ + bias $2$.

## Constraints

- `in_features, out_features ∈ [1, 256]`, `hidden_multiplier ∈ [1, 8]`
- The forward input is a float32 batch of shape `(B, in_features)`
- Evaluation instantiates and checks under a fixed seed (`torch.manual_seed(42)`), so your implementation must guarantee the same result for the same seed - **do not set the seed again in your own code**
