# GradCheck: Reconciling autograd.Function Analytic Gradients with Central Differences

## Problem Description

After writing a custom `autograd.Function`, the only trustworthy verification is comparing its analytic gradient against **finite differences**. Three mistakes recur: obtaining the "analytic" gradient via `torch.autograd.grad(forward(...))` (that validates PyTorch's own backward and never executes the custom one), checking only the first input or only a few elements, and using `eps` itself as the step size (truncation and rounding error explode together). Given `forward, backward, inputs`, return the list of **maximum relative errors**, one per input.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- The analytic gradient **must** go through an `autograd.Function`: `forward` computes the value and saves the inputs, `backward` calls the supplied `backward`, and `torch.autograd.grad` then reads the gradient.
- The numerical gradient uses central differences \((L(x+\delta)-L(x-\delta))/(2\delta)\) for **every element of every input**, with \(\delta_i=\sqrt{eps}\,(1+|x_i|)\) (`eps` is a parameter, defaulting to float64 machine precision).
- Relative error \(rel=|a-n|/\max(|a|,|n|,10^{-8})\); return `list[float]` whose entry \(k\) is the maximum over all elements of `inputs[k]`.
- A non-float64 tensor in `inputs`, a `forward` result that is not 0-dimensional, and a `backward` returning the wrong number of gradients all raise ValueError.

## Example

For \(L=\sum x^2\) the analytic gradient is \(2x\), and the central difference is **exact** for a quadratic (\((x+\delta)^2-(x-\delta)^2=4x\delta\)), leaving only rounding noise:

```
inputs = [ [[1.0, -2.0]] ]
max_rel_error = [ ~1e-16 ]        # asserted < 1e-10
```

## Constraints

- Only float64 inputs are supported (both the step size and the thresholds are designed for float64); `inputs` may hold 1 to 3 tensors of any shape
- Judging pairs "correct backward ⇒ tiny error" with "deliberately wrong backward ⇒ error must exceed the threshold", and uses a three-input case where only the third is wrong to catch skipped inputs
