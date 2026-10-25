# PReLU Explained

## Definition

$$ \text{prelu}(x)=\max(0,x)+a\cdot\min(0,x) $$

That is Leaky-ReLU, except the negative slope $a$ is a **learnable parameter**.

## Why make the slope learnable

- Leaky-ReLU's negative slope is a hyperparameter (e.g. 0.01); PReLU makes it trainable per channel, and Deep PReLU Networks (He et al.) used this to beat ReLU on ImageNet.
- Initializing $a=0.25$ is the value recommended in the paper; $a\to0$ degenerates to ReLU, $a\to1$ degenerates to identity, and $a=0.01$ is Leaky-ReLU.

## Implementation and broadcasting notes

- $a$ usually has shape `(C,)` (one value per channel) and must be broadcast per channel: for $(N,C,H,W)$ reshape it to $(1,C,1,1)$ before multiplying the negative half. This problem implements the broadcasting semantics of the given $a$.
- Decompose it as `torch.where(x>0, x, a*x)` or `relu(x)+a*relu(-x)`; both are equivalent. Note that the output at $x=0$ is always 0.

## Gradients (for understanding)

With respect to the input: 1 for $x>0$, otherwise $a$. With respect to the parameter $a$: the gradient is the sum of the negative-half inputs $\sum_{x<0}x$ — so $a$ is updated in the direction that makes negative values less negative.
