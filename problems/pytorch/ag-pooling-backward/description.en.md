# Pooling Backward: argmax Routing for Max and Window Sharing for Adaptive

## Problem Description

Pooling has no learnable parameters, so the entire difficulty of its backward is **where each gradient goes**: `max` hands all of \(dY\) to the window's largest entry, `avg` spreads it evenly over the window, and `adaptive_avg` uses windows of **unequal, possibly overlapping** length, so a position receives \(dY/\text{that window's real area}\) rather than \(dY/(k_hk_w)\). Three modes, three different divisors and routing rules, all sharing one window-start formula. Given `x, dY` and the mode switches, return `dx`.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- `mode='max'`: give all of \(dY_{b,c,i,j}\) to the maximum entry of the window; on ties take the **smallest row-major index** (top first, then left).
- `mode='avg'`: each position in the window receives \(dY/(k_hk_w)\); `padding` must be 0, otherwise ValueError.
- `mode='adaptive_avg'`: ignore `kernel/stride/padding`; window \(i\) covers \([\lfloor iH/OH\rfloor,\lceil (i+1)H/OH\rceil)\) and the divisor is that interval's actual length.
- Parts of a window outside `x` (the padding region) produce no `dx`; `dx` always has the shape of `x`; an unknown `mode` raises ValueError.

## Example

A \(4\times4\) increasing input with `kernel=2` and `mode='max'`; numbers come from the EXAMPLE_* constants in test_cases.py:

```
x  = [[ 1,  2,  3,  4],    dY = [[1, 2],     argmax of the four windows:
     [ 5,  6,  7,  8],           [3, 4]]     (1,1)=6, (1,3)=8, (3,1)=14, (3,3)=16
     [ 9, 10, 11, 12],
     [13, 14, 15, 16]]
dx = [[0, 0, 0, 0],
      [0, 1, 0, 2],
      [0, 0, 0, 0],
      [0, 3, 0, 4]]
```

Only four entries of `dx` are nonzero; the other 12 are never the window maximum, so they receive nothing.

## Constraints

- `kernel` accepts an int or `(kh, kw)`; `stride=None` defaults to the kernel; `padding` is allowed only for `max`
- Cross-checked against `torch.autograd.grad` of `F.max_pool2d`/`F.avg_pool2d`/`F.adaptive_avg_pool2d`, atol=1e-9 in float64
