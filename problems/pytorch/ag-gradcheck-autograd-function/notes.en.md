# Notes: Why gradcheck Has to Go Through autograd.Function

## Origin

`torch.autograd.gradcheck` is the stock answer to "can you verify a custom operator?" for framework roles, but interviewers more often drill into its **implementation details**: why central differences, why a \(\sqrt{eps}\) step, why every element instead of a few probes. Writing this problem out is equivalent to reimplementing the core sixty lines of gradcheck.

## Balancing Step Size Against Two Error Sources

One-sided differences have truncation error \(O(\delta)\), central differences \(O(\delta^2)\); rounding error is \(O(eps\,|L|/\delta)\) in both cases. With \(\delta=\sqrt{eps}\approx1.5\times10^{-8}\) in float64 the total error of a central difference is about \(O(eps^{3/2})\sim10^{-12}\), the practical optimum. The problem further uses \(\delta_i=\sqrt{eps}(1+|x_i|)\) so that large-magnitude elements are probed proportionally. Dividing by \(\max(|a|,|n|,10^{-8})\) keeps the ratio sane when the gradient is near zero.

## Why the Function Path Is Mandatory

Writing `torch.autograd.grad(forward(*inputs), inputs)` returns PyTorch's automatic differentiation of the forward graph and has nothing to do with the hand-written `backward` that was passed in — the check would pass 100% even if the analytic gradient were entirely wrong. The correct construction wraps both callables in an `autograd.Function` subclass (`forward` computes the value and calls `save_for_backward`, `backward` invokes the user function with `grad_output`), so `torch.autograd.grad` genuinely exercises **the user's backward**.

## Core Ideas and Pitfalls

- **Perturb element by element**: set `flat[i]` to `x+δ` and `x−δ` and run `forward` twice each; perturbing only the first element, or the whole tensor by one δ, misses errors.
- **Check every input**: `errors` must have length `len(inputs)`; the three-input case where only the third gradient is wrong catches loops that run once.
- **Never mutate the arguments**: perturbations happen on `.clone()` copies, so the caller's tensors keep their values and `requires_grad` state.
- **Validate the 0-dim result**: a non-scalar `forward` output must raise ValueError, since `grad_output` is defined as the scalar 1.0 here.
- **Non-differentiable points**: at 0 the analytic gradient of `relu` and its finite difference legitimately differ by a step, so the kink case uses a looser threshold.

## How It Differs from Neighboring Problems

`fw-autograd-function` only asks for the operator itself; this one asks for the **verifier**, switching the perspective from implementing an operator to implementing its test. Problems such as `ag-softmax-backward` provide a gradient formula; here the deliverable is the procedure that decides whether such a formula is right.

## Cross-Check Basis and Complexity

Judging does not use a single oracle but paired cases: a correct backward must produce an error below the threshold (the quadratic demands < 1e-10) and a deliberately wrong backward must exceed it, with the list length and per-entry magnitudes also checked. Complexity is \(O(2\sum_k n_k)\) forward evaluations (two per element), which is why the shapes stay small.
