# Solution Notes · AWQ

## In one sentence: allocate quantization precision to weight channels by "activation importance"

A common misconception in weight quantization is that "all weights are equally important". AWQ's experimental conclusion is the opposite: once the **few weights connected to salient activations** are perturbed by quantization error, the output degrades noticeably. Rather than keeping those channels in FP16 separately (which breaks hardware regularity), it is better to **magnify their weights as a whole** before quantization — magnified, they occupy higher bits in the per-tensor quantizer and their relative error naturally drops. This is the "aware" part of activation-aware Weight Quantization.

## Scaling factor

$$s_j = \frac{\text{x\_bar}_j^\alpha}{\operatorname{mean}_k(\text{x\_bar}_k^\alpha)}$$

The numerator uses the channel's mean activation magnitude raised to the power $\alpha$ to express "how much this channel is used", and the denominator performs arithmetic-mean normalization so that the scaling does not change the overall weight magnitude (otherwise the global scale would drift whenever $\alpha$ changes). $\alpha=0$ degenerates to $s\equiv 1$ (plain RTN), and $\alpha=1$ scales fully by activation size.

## Why choose alpha with the weighted output error

Real AWQ uses a calibration set to compute $\|WX-\hat WX\|_F^2$. This problem uses a reproducible **diagonal approximation** — weighting the squared per-column error by the squared input-channel activation importance $x_{bar,j}^2$, which is exactly the result of taking $H=XX^\top\approx \text{diag}(x_{bar}^2)$. During the grid search, **ties go to the first occurrence** to keep judging deterministic.

## Difference from SmoothQuant (easily confused)

Both compute a "per-input-channel scaling factor", but their direction and purpose differ:

| Aspect | SmoothQuant | AWQ |
|---|---|---|
| What is scaled | **activations divided by s, weights multiplied by s** (both sides move) | **only the weights are scaled, then quantized** (activations unchanged) |
| Purpose | make W8A8 activations quantizable (W is also quantized) | protect the quantization precision of important weight channels (W-only, activations stay FP16) |
| Source of the factor | activation absmax ÷ weight absmax | activation magnitude only (weights do not appear) |

## Identity self-checks

`check_uniform_xbar_neutral`: when all of $x\_bar$ is equal, $s\equiv 1$, every alpha gives the same result, and the first occurrence in the grid, 0.0, should be selected. `check_alpha_uses_importance`: a "large activation, small weight" channel should make a non-zero alpha win.
