# MaxPool2d in Detail

## Definition

It takes the maximum inside each sliding window, producing $(N,C,\Delta)\times$ the number of windows. Purpose: downsampling + translation invariance + keeping the strongest activation.

$$ \text{out}(n,c,i,j)=\max_{(p,q)\in\text{window}} x(n,c,i\cdot s+p,\,j\cdot s+q) $$

## Output size formula (worth memorizing)

$$ H_{out}=\Big\lfloor\frac{H+2p-k}{s}\Big\rfloor+1 $$

With `ceil_mode=True`, replace $\lfloor\cdot\rfloor$ by $\lceil\cdot\rceil$ (one extra row/column; the edges are padded with $-\infty$ before taking the max, which amounts to allowing incomplete windows). stride defaults to kernel.

## The padding pitfall

- The padding for max must be filled with $-\infty$ (or the data minimum); otherwise 0-padding **corrupts** the max (for entirely negative inputs, 0 becomes the maximum instead). Internally torch gives maxpool's pad the $-\infty$ semantics.
- This is also the root cause of the different padding handling between avgpool and maxpool.

## Implementation routes

- Simple and correct: cross-check against `F.max_pool2d(x, kernel_size, stride, padding)`.
- Hand-written: `unfold` the windows and then `.max(dim).values`; for padding use a `masked` value / negative infinity.

## Backward hint (for understanding)

Gradients flow back only to the position that attained the maximum inside the window (argmax routing); all other positions get 0. This is why maxpool preserves "the most salient feature position". avgpool, by contrast, distributes the gradient evenly across all positions of the window.
