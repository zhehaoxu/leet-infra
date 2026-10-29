# LSTM Backward and BPTT: Gate Splitting and Accumulation along Time

## Problem Description

The LSTM backward steps on two rakes at once: the **slice order of the gate vector** and the **accumulation along time**. PyTorch lays out the rows of `W_ih` as `(i, f, g, o)` (input / forget / cell candidate / output); swapping the forget and cell slices is nearly invisible in the forward yet ruins every gradient. Along time, \(dh_t\) must receive both the upstream \(dH_{t+1}\) and the contribution produced by the current step, while \(dW_{ih}=\sum_t dgates_t^T x_t\) accumulates across steps. Given `xs, h0, c0, W_ih, W_hh, b_ih, b_hh, dhT, dcT`, return the seven gradients.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- Forward: \(gates=x_tW_{ih}^T+h_{t-1}W_{hh}^T+b_{ih}+b_{hh}\) sliced as `[0:J]、[J:2J]、[2J:3J]、[3J:4J]` for \(i,f,g,o\) with \(i,f,o=\sigma(\cdot)\) and \(g=\tanh(\cdot)\); \(c_t=f\,c_{t-1}+i\,g\), \(h_t=o\tanh(c_t)\).
- Backward: \(dc_t=d c_{t+1}+dh_t\odot o\odot(1-\tanh^2c_t)\) — that factor \(o\) comes from \(h=o\tanh c\); then \(di=dc\,g,dg=dc\,i,df=dc\,c_{t-1},do=dh\tanh c\); folding into \(dgates\) multiplies sigmoid gates by \(y(1-y)\) and the tanh gate by \(1-y^2\).
- \(db_{ih}=db_{hh}=\sum_t dgates_t\): two identical full-length \(4J\) vectors, never half-length per-gate splits.
- Dimension inconsistencies (row count of `W_ih` not \(4J\), last axis of `xs` not \(E\), …) raise ValueError; compute in float64 and return dtypes matching each input.

## Example

`T=1, N=1, E=2, Hid=1` with zero biases; numbers come from the EXAMPLE_* constants in test_cases.py:

```
x0=[1,1]  h0=[1]  c0=[0]  W_ih row2=[-2,-2]  W_hh row2=[4]  dhT=[1]  dcT=[0]
gates = [0, 0, (-2-2)+4, 0] = [0,0,0,0] ⇒ i=f=o=0.5, g=0 ⇒ c1=0, h1=0
dc1 = 0 + 1*0.5*(1-0) = 0.5 ⇒ dg = 0.5*0.5 = 0.25, other three gates = 0
dgates = [0, 0, 0.25, 0]
dxs=[[-0.5,-0.5]]  dW_ih row2=[0.25,0.25]  dW_hh row2=[0.25]  db=[0,0,0.25,0]
dh0=[1.0]   dc0=[0.25]
```

## Constraints

- `T ≥ 1`, `N ≥ 1`, `Hid ≥ 1`; when both upstream gradients are zero all seven returns must be zero
- Cross-checked against `torch.autograd.grad` of an `nn.LSTM` whose `weight_ih_l0` etc. are `copy_`d in, atol=1e-8 in float64
