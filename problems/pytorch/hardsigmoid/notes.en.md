# Hard-Sigmoid Explained

## Definition

$$ \operatorname{hs}(x)=\operatorname{clip}\!\Big(\frac{x+3}{6},\,0,\,1\Big)=\frac{\operatorname{clip}(x+3,\,0,\,6)}{6}\in[0,1] $$

It approximates the Sigmoid with a polyline (linear on $[-3,3]$, saturating at both ends), avoiding the exponential.

## Design motivation

Sigmoid's $e^x$ is expensive on embedded / fixed-point inference. Hard-Sigmoid needs only one add, one clamp, and one division, which is friendly to INT8 quantization and mobile devices; its output range is $[0,1]$ just like Sigmoid, so it can directly replace it as a gate or activation. It is essentially of the same origin as ReLU6:

$$ \operatorname{HS}(x)=\frac{\operatorname{ReLU6}(x+3)}{6},\qquad \operatorname{HardSwish}(x)=x\cdot\operatorname{HS}(x) $$

## Implementation & Numerical Notes

- The reference solution is `torch.clamp(x + 3.0, min=0.0, max=6.0) / 6.0`: shift first, then clamp, and divide at the end.
- The constants $3$ and $6$ are the **standard convention** (`nn.Hardtanh(-3,3)` normalized to $[0,1]$ is equivalent); do not use the other scale from the MobileNet paper, $(x/2+0.5)$ (with clip points $\pm2.5$) — the two have different slopes on the $|x|<3$ segment and will not match.
- The polyline is non-differentiable at $x=\pm3$; in practice the subgradient is taken as 0 on the saturated segments, and grading does not involve gradients.

## Related

HardSwish = $x\cdot$HardSigmoid; ReLU6 is its upper-bound-clipping close relative; Sigmoid is the function being approximated.
