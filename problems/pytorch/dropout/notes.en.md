# Inverted Dropout Explained

## Definition and "Inverted Scaling"

During training, randomly zero elements with probability $p$, and **divide the retained elements by $1-p$**:

$$ \tilde x_i=\frac{m_i}{1-p}x_i,\qquad m_i\sim\text{Bernoulli}(1-p) $$

This keeps the **expectation of each retained unit unchanged** ($\mathbb E[\tilde x_i]=x_i$), so **no scaling is needed at inference time**—just run the original values forward. This is the default in modern frameworks ("inverted / scalable dropout"), as opposed to the original paper's "no scaling in training, multiply by $1-p$ at inference."

## train vs eval (the core branch)

- `training=True`: sample a mask + divide by $1-p$.
- `training=False` (or $p=0$): **return the identity**, no sampling and no scaling. Getting this wrong throws off all the numbers.

## Reproducibility: Lock the Seed with an Independent Generator (the key to this problem)

Sample with an **independent** `torch.Generator(device=x.device).manual_seed(seed)` and **do not touch the global RNG**:

```
g = torch.Generator(device=x.device).manual_seed(seed)
mask = torch.rand(x.shape, generator=g, device=x.device, dtype=x.dtype) >= p
```

Key points:
1. Keep with `>= p` (probability $1-p$); do not write `> p` or invert the direction.
2. `dtype=x.dtype` and `device=x.device` must follow the input, otherwise half precision/device mismatch occurs.
3. Use an independent generator so that the tests can reproduce the same mask for the same seed; touching the global `torch.manual_seed` would pollute other random operators.

## Relation to Other Regularizers

DropPath, SpatialDropout (zeroing whole channels together), and Variational Dropout (sharing a mask across time steps) are structured variants; the core is always "Bernoulli mask + expectation-preserving scaling."
