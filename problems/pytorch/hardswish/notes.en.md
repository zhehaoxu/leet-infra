# HardSwish Explained

## Definition

$$ \text{hardswish}(x)=x\cdot\frac{\text{ReLU6}(x+3)}{6} $$

where $\text{ReLU6}(t)=\min(\max(t,0),6)$.

## Why it exists

Swish/SiLU requires computing a sigmoid (which involves an exponential), which is unfriendly for mobile / integer-quantized inference. HardSwish approximates Swish with a **piecewise-linear** function, removing the exponential and saving hardware cost; it is the activation of MobileNetV3.

$$ \text{hardswish}(x)\approx \text{swish}(x), $$

but only on $[-3,3]$ is it a polyline with gradually changing slope; at the two ends: $x\le-3\to0$, $x\ge3\to x$.

## Implementation notes (the boundaries are the easiest place to go wrong)

- Clip $x+3$ to $[0,6]$ and then divide by 6, which is equivalent to the gate $g(x)$:
  - $x<-3$: $g=0$, output 0
  - $-3\le x\le 3$: $g=(x+3)/6$, output $x(x+3)/6$
  - $x>3$: $g=1$, output $x$
- Use `x * torch.clamp(x+3, 0, 6) / 6`. Note the denominator is 6; do not write the other constant version found in `F.hardswish` (PyTorch's official `F.hardswish` uses the gate $\text{ReLU6}(x+3)/6$, consistent with this problem).

## Relation to HardSigmoid

HardSigmoid $=\text{ReLU6}(x+3)/6$ is exactly the gate here; HardSwish $=x\cdot$HardSigmoid. The two differ by a factor of $x$.
