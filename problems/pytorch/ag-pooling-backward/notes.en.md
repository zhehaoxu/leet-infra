# Notes: Routing and Divisors for the Three Pooling Modes

## Origin

The pooling backward shows up in written tests for vision/multimodal roles and also serves as a hook for "what do you actually understand autograd to be": the gradient of `max` is a **subgradient at a non-differentiable point**, while `adaptive_avg` tests whether you can reproduce PyTorch's `start_index = floor(i·H/OH)` window definition exactly. Inference-engineering candidates must be able to write it, because exporting to ONNX expands these two by hand.

## max: Routing and the Tie Convention

Since \(\partial y_{ij}/\partial x_{pq}=\mathbb 1[(p,q)=\arg\max_{\text{window}}x]\), \(dx\) is "scatter \(dY\) back through the argmax", implemented either by recording forward indices (`return_indices=True`) or by re-comparing inside the window. Ties are not consistent between PyTorch's CPU and CUDA kernels, so this problem fixes **smallest row-major index** (the first maximum, as `argmax` returns) to keep judging deterministic. When using `scatter`, flatten the indices to 1-D positions of shape `(B,C,OH,OW)`; a wrong `dim` sprays gradients into other channels.

## Divisors: avg versus adaptive_avg

For fixed-window `avg` the divisor is always \(k_hk_w\) (with `padding=0`), whereas in adaptive pooling the window height \(w_i=\lceil(i+1)H/OH\rceil-\lfloor iH/OH\rfloor\) varies with \(i\) and windows **overlap** whenever \(OH\nmid H\), so one position can belong to two windows and accumulates two shares \(dY/w_iw_j\). "Always divide by the kernel area" fails on such inputs, and the test includes shapes like \(H=5,OH=2\).

## Core Ideas and Pitfalls

- **`stride` separate from `kernel`**: the default is `stride=kernel`, but a smaller explicit stride makes windows overlap and both the argmax routing and the average accumulation must follow `stride`.
- **`padding` only for `max`**: padded positions have no corresponding element of \(x\) and the divisor of `avg` becomes ambiguous, so this problem raises ValueError instead.
- **dx always has the shape of x**: never return a de-padded size.
- **dtype**: even an integer `x` must come back as floats matching the input dtype, with float64 internals.

## How It Differs from Neighboring Problems

`ag-conv2d-backward` solves for both \(dx\) and a learnable \(dw\); `adaptive-avgpool` and `adaptive-maxpool` in this bank are **forward** problems that only compute outputs and window bounds. Here only the backward routing is in play.

## Cross-Check Basis and Complexity

The oracle runs `F.max_pool2d`, `F.avg_pool2d` and `F.adaptive_avg_pool2d` forward and then `torch.autograd.grad((y*dY).sum(), x)`, atol=1e-9 in float64. The example uses a strictly increasing \(4\times4\) so the four argmax positions are obvious by inspection. Complexity is \(O(BCHW\cdot k_hk_w)\) by window enumeration, or \(O(BCHW)\) by index scattering.
