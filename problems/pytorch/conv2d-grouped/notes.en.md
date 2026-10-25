# Grouped Convolution in Depth

## Channel partitioning and "group isolation"

In a standard convolution, every output channel looks at **all** `Cin` input channels; grouped convolution cuts `Cin` into `groups` segments,
and the output channels of segment `g` **only look at the input channels of their own segment**:

$$ \text{out}[\,g{:}o\,] = \sum_{c \,\in\, \text{group } g} \text{conv}(x_{c},\, w_{g,o,c}) $$

Hence the input-channel dimension of `weight` keeps only $C_{in}/\text{groups}$. This brings parameters and compute down by $1/\text{groups}$,
at the cost that groups no longer exchange information — re-mixing must rely on a subsequent 1×1 or on the network architecture (e.g. ShuffleNet's channel shuffle).

## Two degenerate cases

- `groups == 1` → an ordinary convolution.
- `groups == Cin` → **depthwise convolution**, with one independent kernel per channel and weight view `(Cin, 1, kh, kw)`.
  This is exactly the boundary that this problem's "group locality" probe detects: perturbing channel 0 (in group 0) must never change group 1's output.

## Common mistakes

- **Slice order**: both input and output channels are split **contiguously** by group; don't reverse `weight`'s dimension order
  (it is `(Cout, Cin//g, kh, kw)`, not `(Cin//g, Cout, ...)`, which is the transposed-convolution convention).
- `Cin` and `Cout` must both be divisible by `groups`, otherwise the reshape fails.
- The bias is still added over the full `Cout` and is unaffected by grouping.

## Applications

ResNeXt uses `cardinality` (i.e. groups) to raise capacity at equal compute; MobileNet/ShuffleNet combine `groups=Cin`
depthwise convolution with pointwise to form "depthwise separable convolution", reducing the $3\times3$ dense convolution's $C_{in}C_{out}kk$
to $C_{in}kk + C_{in}C_{out}$ — the cornerstone of mobile CNNs.
