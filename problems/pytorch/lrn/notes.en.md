# LRN in Detail

## Cross-Channel Lateral Inhibition

The core idea of LRN is that neighbouring channels inhibit each other: the larger a channel's activation at some spatial position,
the more the sum of squares of the channels around it (same position, adjacent channels) divides its response down. The formula is

$$ y_{i} = \frac{x_{i}}{\left(k + \dfrac{\alpha}{\text{size}} \sum_{j \in \text{win}(i)} x_{j}^{2}\right)^{\beta}} $$

The window $\text{win}(i)$ runs along the **channel dimension**, is symmetric about $i$ and has width `size`; out-of-range channels are zero-padded. Note that the
"local" in this normalization means **local in channels**, not local in space - for each $(h,w)$ it slides independently along $C$.

## Lineage Relative to BN / LN

- **BatchNorm**: standardizes each channel using batch x space statistics (with learnable $\gamma,\beta$); inference depends on running stats.
- **LayerNorm**: zero-mean unit-variance standardization over the feature dimension (channels and so on), independent of the batch; commonly used by Transformers.
- **LRN**: neither subtracts the mean nor divides by the standard deviation; it only "divides by a power of the neighbouring channels' sum of squares", which is a form of **mean-free**,
  data-dependent gain suppression. It has no learnable parameters and needs no batch statistics.

Precisely because LRN brings limited gains while slowing training, LeCun's later work and the networks after AlexNet widely abandoned it in favour of BN.

## Common Mistakes

- **Window direction**: the window runs along the channel dimension (dim=1), not the spatial dimensions; the $H,W$ elements are mutually independent.
- **`alpha/size`, not `alpha`**: the sum of squares must be divided by the window width `size` before multiplying by `alpha`, otherwise it does not
  match `F.local_response_norm` (which is why this problem's dumped reference solution deliberately keeps the `/size`).
- **Boundary zero padding**: the first/last channels' windows are shorter than `size`, and missing terms count as $0$ in the sum of squares (equivalent to zero padding);
  do not use periodic or replicated extension.
- `size` must be **odd** for the window to have a symmetric center; the sum in the denominator **includes the channel itself** $x_i^2$.

## Applications and History

LRN came from Krizhevsky's AlexNet (2012), where it normalized large ReLU activations to stabilize training. But Sainath
et al. (2012) and later empirical work showed that its gains are negligible on most modern architectures, and after BatchNorm (2015) appeared it largely left the mainstream.
Today it mainly serves as a teaching contrast within the "normalization family" and as an example for understanding the biologically inspired origin of lateral inhibition.
