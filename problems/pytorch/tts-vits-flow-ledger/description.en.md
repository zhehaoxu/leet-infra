# VITS Normalizing Flow Forward/Reverse and log-det Ledger

## Background

VITS is an end-to-end speech synthesis model. A large part of its acoustic model is a **normalizing flow**: it maps a sequence of Gaussian noise through a series of invertible transformations into acoustic latents, and uses the inverse of those transformations to map latents back to noise. At training time the likelihood is computed from the difference of the joint densities before and after the transform, so every invertible transformation must report its **log-det** (logarithm of the determinant) in addition to its output.

All of these flows live in `modules.py` and `models.py` of the VITS repository:

- `Log`: element-wise logarithm (with a lower-bound clamp).
- `Flip`: reverses the channel dimension.
- `ElementwiseAffine`: per-channel affine transform.
- `ResidualCouplingLayer`: splits channels in half, passes one half through and applies an affine to the other (coupling).
- `ResidualCouplingBlock`: chains several couplings and flips into a flow sequence.
- `StochasticDurationPredictor`: a differentiable duration predictor whose NLL contains both an `nll` and a `logq` ledger.

This task asks you to rebuild these ledgers in pure Python and to nail down several details of the official implementation that **you will get wrong unless you read the source**.

## Conventions

- Everything that can be expressed as an exact rational number (internally via `fractions.Fraction`) must go through `Fraction`; `log` / `exp` are irrational and are computed as `float`. **All returned values are converted to `float`.**
- A three-dimensional tensor has shape `(B, C, T)`: `B` is batch, `C` is channel, `T` is time steps.
- `x` is a nested list and is flattened to one dimension following `(B, C, T)` before use; the flattened length must equal `B*C*T`, otherwise an error is raised.
- The mask is called `x_mask` officially. The official code requires two different shapes:
  - the `x_mask` used by coupling / block has shape `(B, 1, T)` and **broadcasts** along the channel dimension;
  - `Log` / `ElementwiseAffine` require the mask to have the **same shape** as `x` (`(B, C, T)`).

  This task accepts both shapes: if the flattened mask has length `B*T` it is broadcast from `(B, 1, T)` to `(B, C, T)`; if the length is `B*C*T` it is used directly as `(B, C, T)`. If neither matches, an error is raised.
- `m` / `logs` are **per-channel** parameters: a scalar (broadcast to all channels), a one-dimensional list of length `C`, or a nested list of shape `(C, 1)`.
- Every invalid input raises `ValueError` (for example `shape` not being three positive integers, an `x` length mismatch, an odd channel count for coupling, an `m` length mismatch, and so on).
- `reverse=True` selects the official reverse branch.

## mode 1: `flow_primitive` -- a single flow primitive

```python
solve("flow_primitive", kind="log", shape=[1, 1, 3],
      x=[[[0.5, 2.0, 4.0]]], mask=[[[1.0, 1.0, 1.0]]])
```

Signature:

```python
solve("flow_primitive", kind, shape, x, mask, m=None, logs=None, reverse=False)
```

Returns:

```python
{"y": [...], "logdet": [...]}   # y is flattened to B*C*T; logdet has one entry per batch
```

`kind` has three options:

### `kind="log"`

Official `modules.py::Log.forward`:

```python
def forward(self, x, x_mask, reverse=False, **kwargs):
    if not reverse:
      y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
      logdet = torch.sum(-y, [1, 2])
      return y, logdet
    else:
      x = torch.exp(x) * x_mask
      return x
```

Key points:

- The forward pass is `y = log(clamp_min(x, 1e-5)) * mask`: **clamp to `1e-5` first, then take the logarithm**, and finally multiply by the mask as a whole.
- `logdet = sum(-y)`, summed over the channel and time dimensions with one entry per batch. **The mask has already been folded into `y`**, so masked-out positions contribute `0` to the log-det.
- The reverse pass is `x = exp(x) * mask`: **the reverse pass also multiplies by the mask**.

### `kind="flip"`

Official `modules.py::Flip.forward`:

```python
def forward(self, x, *args, reverse=False, **kwargs):
    x = torch.flip(x, [1])
    if not reverse:
      logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
      return x, logdet
    else:
      return x
```

Key points:

- ★★ It flips the **channel axis** (`torch.flip(x, [1])`), **not** the time axis. An input of shape `(1, 2, 3)` comes out as `(3, 2, 1)`.
- ★ `Flip.forward` **completely ignores `x_mask`** (there is not even a mask parameter in the signature); positions whose mask is 0 are flipped just the same.
- `logdet` is always `0`; the reverse pass is also a flip (self-inverse).

### `kind="affine"`

Official `modules.py::ElementwiseAffine.forward`:

```python
def forward(self, x, x_mask, reverse=False, **kwargs):
    if not reverse:
      y = self.m + torch.exp(self.logs) * x
      y = y * x_mask
      logdet = torch.sum(self.logs * x_mask, [1,2])
      return y, logdet
    else:
      x = (x - self.m) * torch.exp(-self.logs) * x_mask
      return x
```

Key points:

- Forward is `y = (m + exp(logs) * x) * mask`: **the mask applies to the whole expression**, so masked-out positions output `0`.
- `logdet = sum(logs * mask)`, summed over channel and time with one entry per batch.
- Reverse is `x = (x - m) * exp(-logs) * mask`: **the mask applies to the whole expression here as well**.
- Officially `m` / `logs` are `nn.Parameter(torch.zeros(channels, 1))`, i.e. of shape `(channels, 1)`.

## mode 2: `coupling_layer` -- a single coupling layer

```python
solve("coupling_layer", shape=[1, 4, 2],
      x=[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]],
      mask=[[[1.0, 1.0]]], m=[[[0.1, 0.2], [0.3, 0.4]]])
```

Signature:

```python
solve("coupling_layer", shape, x, mask, m, logs=None, mean_only=True, reverse=False)
```

Official `modules.py::ResidualCouplingLayer.forward` (this task replaces `pre`/`enc`/`post` with directly supplied `m` / `logs`):

```python
def forward(self, x, x_mask, g=None, reverse=False):
    x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
    h = self.pre(x0) * x_mask
    h = self.enc(h, x_mask, g=g)
    stats = self.post(h) * x_mask
    if not self.mean_only:
      m, logs = torch.split(stats, [self.half_channels] * 2, 1)
    else:
      m = stats
      logs = torch.zeros_like(m)
    if not reverse:
      x1 = m + x1 * torch.exp(logs) * x_mask
      x = torch.cat([x0, x1], 1)
      logdet = torch.sum(logs, [1, 2])
      return x, logdet
    else:
      x1 = (x1 - m) * torch.exp(-logs) * x_mask
      x = torch.cat([x0, x1], 1)
      return x
```

Key points:

- ★ The channel count must be even (officially `assert channels % 2 == 0`), and `half = C // 2`.
- `x0` is the first half of the channels and is **passed through unchanged** (the official code does not mask `x0`); `x1` is the second half.
- The length of `m` must be `B * half * T`.
- With `mean_only=True` (how the official block uses it) `logs` is always `0`, so the log-det is always `0`.
- With `mean_only=False` you must supply `logs`, whose length equals that of `m`.
- ★★ The forward pass is `x1 = m + x1 * exp(logs) * x_mask`: **the mask multiplies only the `x1` term, `m` is not masked** → at a position whose mask is 0 the output is **`m`** (not `0`).
- ★★ `logdet = sum(logs, [1, 2])`: **it is not multiplied by the mask**. Even where the mask is 0, the `logs` at those positions still count toward the log-det.
- The reverse pass is `x1 = (x1 - m) * exp(-logs) * x_mask` (the mask applies to the whole expression).

## mode 3: `coupling_block` -- the flow orchestration ledger

```python
solve("coupling_block", shape=[1, 4, 2],
      x=[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]],
      mask=[[[1.0, 1.0]]],
      m_list=[[[[0.1, 0.2], [0.3, 0.4]]], [[[0.5, -0.25], [0.125, 0.75]]]])
```

Signature:

```python
solve("coupling_block", shape, x, mask, m_list, reverse=False)
```

Official `models.py::ResidualCouplingBlock`:

```python
self.flows = nn.ModuleList()
for i in range(n_flows):
  self.flows.append(modules.ResidualCouplingLayer(..., mean_only=True))
  self.flows.append(modules.Flip())

def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)
    return x
```

Key points:

- The flow sequence is fixed as `[Coupling_0, Flip, Coupling_1, Flip, ...]`, of length `2 * len(m_list)`.
- ★ The forward pass executes the flows in **forward** order and the reverse pass in **reverse** order (`reversed(self.flows)`). The order cannot be swapped: coupling and flip do not commute.
- Every coupling uses `mean_only=True` → `logs` is always `0` → **the log-det of the whole block is always `0`**.
- ★ Officially the forward pass discards the log-det (`x, _ = flow(...)`), and the reverse branch even returns only `x`. This task uniformly returns the accumulated log-det (all zeros).
- `m_list` must not be empty.

## mode 4: `duration_nll` -- the duration predictor's NLL and logq ledgers

```python
solve("duration_nll", w=[[[2.0, 3.0, 4.0]], [[1.5, 2.5, 0.5]]],
      e_q=[[[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]],
            [[0.7, 0.8, -0.9], [1.0, 0.2, -0.3]]],
      z_u=[[[0.25, -0.5, 0.75]], [[0.1, 0.2, 0.3]]],
      z1=[[[0.11, -0.22, 0.33]], [[0.44, -0.55, 0.66]]],
      mask=[[[1.0, 1.0, 1.0]], [[1.0, 0.0, 1.0]]],
      post_logdets=[0.5, -0.25], flow_logdets=[1.25, 0.75])
```

Signature:

```python
solve("duration_nll", w, e_q, z_u, z1, mask, post_logdets, flow_logdets)
```

Official `models.py::StochasticDurationPredictor.forward` (this task replaces the random quantities and the internal flow structure with supplied values):

```python
logdet_tot_q = 0
e_q = randn(B,2,T) * x_mask
z_q = e_q
for flow in self.post_flows:
  z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
  logdet_tot_q += logdet_q
z_u, z1 = torch.split(z_q, [1, 1], 1)
u = torch.sigmoid(z_u) * x_mask
z0 = (w - u) * x_mask
logdet_tot_q += sum((logsigmoid(z_u) + logsigmoid(-z_u)) * x_mask, [1, 2])
logq = sum(-0.5 * (log(2*pi) + e_q**2) * x_mask, [1, 2]) - logdet_tot_q

logdet_tot = 0
z0, logdet = self.log_flow(z0, x_mask)
logdet_tot += logdet
z = torch.cat([z0, z1], 1)
for flow in flows:
  z, logdet = flow(z, x_mask, g=x, reverse=reverse)
  logdet_tot = logdet_tot + logdet
nll = sum(0.5 * (log(2*pi) + z**2) * x_mask, [1, 2]) - logdet_tot
return nll + logq
```

Meaning of each input:

- `w`: shape `(B, 1, T)`, the raw duration.
- `e_q`: shape `(B, 2, T)`, the Gaussian noise (officially `randn * x_mask`).
- `z_u` / `z1`: each of shape `(B, 1, T)`, the two halves that `z_q` is split into after `post_flows`.
- `mask`: shape `(B, 1, T)`.
- `post_logdets` / `flow_logdets`: each a list of length `B`, holding the sum of the log-dets of `post_flows` and of `flows` respectively.

Returns:

```python
{
  "logq": [...], "nll": [...], "total": [...],
  "logdet_tot_q": [...], "logdet_tot": [...],
  "z0": [...], "z0_log": [...],
}
```

Key points:

- ★★ `logq = gauss - logdet_tot_q`, where `logdet_tot_q = post_logdets[b] + lsig` and `lsig = sum((logsigmoid(z_u) + logsigmoid(-z_u)) * mask)`.
  **`lsig` is subtracted** (it is itself negative, so `logq` actually grows). Writing `gauss + lsig - post` is wrong.
- `gauss = sum(-0.5 * (log(2*pi) + e_q**2) * mask)`, summed over both channels of `e_q` and over the time steps.
- ★ `u = sigmoid(z_u) * mask` and `z0 = (w - u) * mask`: **`z0` is multiplied by the mask as a whole**.
- `log_flow` is the `Log` primitive: `z0_log = log(clamp_min(z0, 1e-5)) * mask` and `logdet_log = sum(-z0_log)`.
- `logdet_tot[b] = logdet_log[b] + flow_logdets[b]`.
- `z = cat([z0_log, z1])`, `nll = sum(0.5 * (log(2*pi) + z**2) * mask) - logdet_tot`, and `total = nll + logq`.

## Common mistakes

1. Flipping `Flip` along the time axis (`torch.flip(x, [2])`) -- officially it is `torch.flip(x, [1])`, the channel axis.
2. Multiplying `Flip` by the mask -- the official `Flip.forward` has no mask parameter at all.
3. Getting the order of `Log` wrong (multiplying by the mask before clamping instead of clamping to `1e-5` first).
4. Forgetting the mask in the reverse branch of `Log` -- officially it is `torch.exp(x) * x_mask`.
5. Writing the log-det of `Log` as `sum(y)` instead of `sum(-y)`.
6. Forgetting the mask in the log-det of `ElementwiseAffine` (officially it is `sum(self.logs * x_mask, [1,2])`).
7. Writing the reverse of `ElementwiseAffine` as `(x + m)` instead of `(x - m)`.
8. Applying the mask to the whole expression in the forward pass of coupling (`(m + x1*exp(logs)) * mask`) -- officially it multiplies only the `x1` term.
9. Multiplying the log-det of coupling by the mask -- officially `sum(logs, [1,2])` does not.
10. Masking or transforming `x0` in coupling -- officially `x0` is passed through unchanged.
11. Still using the supplied `logs` when `mean_only=True` -- officially `logs = zeros_like(m)` in that case.
12. Executing the block's flows in forward order during the reverse pass -- officially it is `reversed(self.flows)`.
13. Placing `Flip` before the coupling in the block -- officially the coupling comes first and the flip second.
14. Using a plus sign for the `lsig` term of `logq` -- it is subtracted.
15. Forgetting to multiply `z0` by the mask as a whole.
16. Only accounting for one of the two halves (`z0_log` and `z1`) in the `nll`.

## Self-check

- `mode 1` `flip`: given `x = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]`, the first two output elements should be `4.0` and `6.0` (the channels are swapped).
- `mode 1` `log` reverse: given a mask that is 0 at some position, the output at that position should be `0`.
- `mode 2`: given a mask that is 0 at some `x1` position, the output at that position should be `m` (not `0`).
- `mode 2`: the `logs` at a position whose mask is 0 still count toward the log-det.
- `mode 3`: the reverse of a block with a single coupling must be "flip first, then the reverse coupling".
- `mode 4`: `logq` is always greater than `gauss - post_logdets` (because a negative number was subtracted).
