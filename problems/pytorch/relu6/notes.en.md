# ReLU6 Explained

## Why put an upper bound on ReLU

ReLU output is unbounded above, so activations can spread over a very large range, and int8 quantization then has to reserve a very wide
scale for "occasional extreme values", diluting precision. ReLU6 pins each layer's output to $[0,6]$, so the quantization grid only needs to cover this small interval, giving a higher
effective bit resolution and better suited to mobile fixed-point inference — exactly why Howard et al. 2017 chose it in "MobileNets".

$$ \operatorname{ReLU6}(x)=\min(\max(0,x),6) $$

## Relation to ReLU / clipped

- `relu6 == torch.clamp(x, 0, 6)`, i.e. ReLU wrapped in another layer of `min(·,6)`.
- The upper bound "6" is an empirical hyperparameter (generalizable to `HardTanh` with an arbitrary `max_value`), not a mathematical necessity.
- Gradient: 1 where $0<x<6$, 0 where $x\le0$ or $x\ge6$ — both ends are "dead zones", and the saturated side can also lose information
  like ReLU when large activations are truncated.

## Successors: ReLU6 → Swish/HardSwish

MobileNetV3 replaces ReLU6 with **Swish / HardSwish** for better accuracy; HardSwish is also essentially "an approximate bounded activation with lower bound
0 and upper bound 3" (see the swish / hardswish problems in this collection). ReLU6 is the starting point for understanding this evolution of "activations designed for quantization".

## Implementation notes

An elementwise `clamp` suffices; be sure to preserve the input dtype (float32/float64) and shape. It is stateless and needs no branching,
and on hardware it is a single fused min/max instruction, extremely cheap.
