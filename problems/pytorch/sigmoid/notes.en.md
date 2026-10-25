# Sigmoid Explained

## Definition

$$ \sigma(x)=\frac{1}{1+e^{-x}}\in(0,1),\qquad \sigma'(x)=\sigma(x)\,(1-\sigma(x)) $$

It squashes any real number into $(0,1)$; the curve is an S shape symmetric about the origin, with $\sigma(-x)=1-\sigma(x)$. Historically it was used for binary-classification output layers and as the switch inside "gates".

## Implementation and numerical details

The naive `1/(1+torch.exp(-x))` makes $e^{-x}$ **overflow to inf** when $x$ is very negative; the result still tends to 0 but comes with overflow warnings and a risk of NaN. The reference solution branches on the sign so the exponent argument is always non-positive:

- $x\ge 0$: $\dfrac{1}{1+e^{-x}}$ (the exponent holds $-x\le0$).
- $x<0$: $\dfrac{e^{x}}{1+e^{x}}$ (the exponent holds $x<0$).

The two expressions are algebraically identical; only the overflowing branch is rewritten. `torch.sigmoid` is implemented this way internally.

## Vanishing gradients

Saturation sets in once $|x|$ is moderately large: as $\sigma\to0/1$ the derivative $\to0$ (its peak is only 0.25 at $x=0$). Stacking sigmoids in a deep network flattens the gradient — this is exactly why ReLU replaced sigmoid in hidden layers. Today sigmoid mostly lives in **gating** (the LSTM $i,f,o$ gates, GRU, attention soft-capping) and probability outputs.

## Related

tanh $=2\sigma(2x)-1$; Hard-Sigmoid is its cheap piecewise-linear approximation; Gumbel-Softmax, LoRA scaling and others use sigmoid as a building block.
