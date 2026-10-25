# Alignment Training: KTO Prospect-Theoretic Loss

## Problem

**KTO** (Kahneman-Tversky Optimization, Ethayarajh et al., arXiv:2402.01306) borrows from prospect theory: people judge outcomes as "gains/losses" relative to a **reference point** \(z_0\) and are more sensitive to losses. It needs only binary good/bad labels (no paired preferences).

Given each sample's **implicit log-ratio** \(z = \log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}\) (sequence-level) and its label (desirable / undesirable), the value function is measured in sigmoid form relative to the reference point \(z_\text{ref}\):

\[ v = \begin{cases} \sigma\bigl(\beta(z - z_\text{ref})\bigr), & y\ \text{desirable}\\[4pt] \sigma\bigl(\beta(z_\text{ref} - z)\bigr), & y\ \text{undesirable}\end{cases} \]

Minimize the "unrealized value", weighting good/bad samples by \(\lambda_D,\lambda_U\) respectively:

\[ \mathcal{L}_i = \begin{cases} \lambda_D\,\bigl(1 - \sigma(\beta(z - z_\text{ref}))\bigr), & \text{desirable}\\ \lambda_U\,\bigl(1 - \sigma(\beta(z_\text{ref} - z))\bigr), & \text{undesirable}\end{cases},\qquad \mathcal{L}=\frac1N\sum_i \mathcal{L}_i \]

The reference point \(z_\text{ref}\) (in the paper estimated from the KL and detached as a constant) is given here as an explicit input.

## Rules

Inputs are `log_ratios` (\((N,)\), i.e. \(z\)), `labels` (\((N,)\), 1=desirable, 0=undesirable), and the scalars `z_ref`, `beta`, `lambda_D`, `lambda_U`. Return a 0-dim float32 scalar = the mean KTO loss. Do not use an off-the-shelf alignment library.

## Example

`log_ratios=[0.5,-0.3,0.2,-0.8]`, `labels=[1,0,1,0]`, `z_ref=0.1`, `beta=0.1`, `lambda_D=lambda_U=1.0`:

```
i0 (good): 1-σ(0.1*(0.5-0.1))=1-σ(0.04)=0.490000
i1 (bad): 1-σ(0.1*(0.1-(-0.3)))=1-σ(0.04)=0.490000
i2 (good): 1-σ(0.1*(0.2-0.1))=1-σ(0.01)=0.497500
i3 (bad): 1-σ(0.1*(0.1-(-0.8)))=1-σ(0.09)=0.477502
loss = mean = 0.488754
```

## Constraints

- CPU float32; \(\beta,\lambda_D,\lambda_U>0\); `labels` take 0/1.
- desirable uses \(\sigma(\beta(z-z_\text{ref}))\), undesirable uses \(\sigma(\beta(z_\text{ref}-z))\), the two directions are opposite (a reflection of loss aversion).
- When \(z=z_\text{ref}\) the value \(=\sigma(0)=0.5\) and the per-sample loss \(=\lambda/2\).
