# GroupNorm Explained

## Definition

Split the channels into $G$ groups and normalize the channels within each group together:

$$ \hat{x}=\frac{x-\mu_g}{\sqrt{\sigma_g^2+\epsilon}},\qquad y=\gamma x+\beta\ (\text{per channel}) $$

$\mu_g,\sigma_g$ are computed within each sample and each channel group.

## Position: a compromise between BN and LN

- BN computes statistics along the batch (across samples) → depends on the batch size and performs poorly for small batches; LN works along features (per sample) → suited to sequences.
- **GN computes per-sample statistics along channel groups**: independent of the batch, friendly to small batches / detection and segmentation, and it is the choice of Stable Diffusion and most large vision models.
- Special cases under a unified view: $G=1$ ≈ the global version of InstanceNorm over the whole image; $G=C$ (one channel per group) degenerates to **InstanceNorm**. GN is a continuous knob between BN and IN.

## The reshape trick (the most elegant implementation)

After $(B,C,H,W)\to(B,G, C/G\cdot H\cdot W)$, compute $\mu,\sigma$ over the last dimension (`dim=-1, keepdim`), normalize, and reshape back to $(B,C,H,W)$. Reshape $\gamma,\beta$ of shape $(C,)$ to $(1,C,1,1)$ for broadcasting.

## Pitfalls

- Do not compute statistics along the batch dimension — GN ignores N entirely.
- It is only meaningful when `C % G == 0`; different group counts give different results, so use the G given in the problem.
- eps goes inside the square root.

## Cross-check reference solution

`F.group_norm(x, num_groups, weight, bias, eps)`.
