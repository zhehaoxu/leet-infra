# Depthwise Separable Convolution in Depth

## Why splitting is cheaper

A standard $3\times3$ convolution has compute $\propto C_{in}\cdot C_{out}\cdot k^2$. Split into two steps:

- **depthwise**: each of the $C_{in}$ channels uses one $k\times k$ kernel $\Rightarrow C_{in}\cdot k^2$;
- **pointwise**: a $1\times1$ dense mixing $\Rightarrow C_{in}\cdot C_{out}$.

The total is $C_{in}(k^2 + C_{out})$, about $k^2$ times less than a dense convolution (with $k=3$, roughly 8–9 times fewer parameters and FLOPs).
MobileNet's "width multiplier" precisely scales down the channels of these two steps.

## Key points: depthwise does not cross channels, pointwise does not cross space

- The depthwise weight has shape $(C,1,k,k)$, equivalent to a convolution with `groups=C`; **channels are fully isolated**,
  which can be verified using only the $1\times1$ weights of one row in `pw` (perturbing other channels should not change the output).
- The pointwise weight is $(C_{out},C,1,1)$: a pure channel linear combination that leaves the $3\times3$ spatial relationships unchanged.
- The order of the two cannot be swapped; without pointwise, the channel count could never change and channels would never interact.

## Preserving the size and odd kernels

The problem uses `padding=kh//2` and `stride=1` so that the depthwise output has the same size as the input (`same` semantics).
Hence **kh is required to be odd** — with an even kernel, `kh//2` cannot make the four sides symmetric and the size would be off.

## Extensions

The depthwise separable idea also reaches Transformers: relative positional attention splits "position-wise convolution" into per-head depthwise +
$1\times1$; separable conv is likewise the basic building block of many real-time/mobile backbones (MobileNetV1→V3, Xception, EfficientNet).
