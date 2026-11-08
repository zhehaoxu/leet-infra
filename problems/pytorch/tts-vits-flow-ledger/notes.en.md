# VITS Normalizing Flow Source Anatomy

> Official repository: `jaywalnut310/vits` (`main` branch)
> Files involved: `modules.py`, `models.py`
> Classes involved: `Log`, `Flip`, `ElementwiseAffine`, `ResidualCouplingLayer`, `ResidualCouplingBlock`, `StochasticDurationPredictor`

This article walks through the official implementation line by line, flags the places **you will get wrong unless you read the source**, and explains how the judgement for this task is designed.

## 1. The overall structure of the official flows

Both the acoustic model and the duration predictor of VITS use the same "invertible transform plus log-det" paradigm:

```
z ~ N(0, I)
  -> post_flows (coupling + flip alternating)      # yields z_q
  -> split into z_u / z1
  -> u = sigmoid(z_u); z0 = (w - u)
  -> log_flow (the Log primitive)
  -> flows (coupling + flip alternating)            # yields z
```

The training-time likelihood is `nll + logq`, where:

- `nll` is the standard normal negative log-likelihood of `z` **after** the transform (minus the log-det);
- `logq` is the negative log-likelihood of `z_q` **before** the transform (minus the log-det).

The log-dets of the two paths have opposite signs, and together they give the "volume change" of the flow.

## 2. ★★ `Log`: the details of clamp and log-det

Official `modules.py`:

```python
class Log(nn.Module):
  def forward(self, x, x_mask, reverse=False, **kwargs):
    if not reverse:
      y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
      logdet = torch.sum(-y, [1, 2])
      return y, logdet
    else:
      x = torch.exp(x) * x_mask
      return x
```

Three key points:

1. **`clamp_min(x, 1e-5)` comes before the multiplication by the mask.** The order cannot be swapped: if you multiply by the mask first and then clamp, positions whose mask is 0 get clamped to `1e-5` and yield `log(1e-5) != 0`.
2. **`logdet = torch.sum(-y, [1, 2])`** -- the two summed dimensions are channel and time (not batch), yielding one entry per batch. The mask is already folded into `y`, so masked-out positions contribute 0.
3. ★ **The reverse branch is `x = torch.exp(x) * x_mask`, which also multiplies by the mask.** And the reverse branch **returns only `x`**, no log-det (many official flows look like this in their reverse branch).

## 3. ★★ `Flip`: it flips the channel axis

Official `modules.py`:

```python
class Flip(nn.Module):
  def forward(self, x, *args, reverse=False, **kwargs):
    x = torch.flip(x, [1])
    if not reverse:
      logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
      return x, logdet
    else:
      return x
```

Three key points:

1. ★★ **`torch.flip(x, [1])` flips dimension 1.** In the `(B, C, T)` layout, dimension 1 is the **channel** axis. Many people write `[2]` (the time axis) here; measured, an input of shape `(1, 2, 3)` comes out as `(3, 2, 1)` rather than a time-reversed `(1, 2, 3)`.
2. ★ **The signature is `forward(self, x, *args, reverse=False, **kwargs)`** -- there is not even an `x_mask` parameter, so **`Flip` completely ignores the mask**. This is not "the mask is treated as all ones" but "the mask never took part in the computation at all".
3. `logdet` is always 0 (the absolute value of the determinant of a flip is 1, so its log is 0). The reverse branch also returns `x`, and both forward and reverse are a flip (self-inverse).

## 4. `ElementwiseAffine`: parameter shapes and the whole-expression mask

Official `modules.py`:

```python
class ElementwiseAffine(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.channels = channels
    self.m = nn.Parameter(torch.zeros(channels,1))
    self.logs = nn.Parameter(torch.zeros(channels,1))

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

Three key points:

1. ★ `m` / `logs` are **`nn.Parameter(torch.zeros(channels, 1))`**, of shape `(channels, 1)` rather than `(channels,)`. This task relaxes that to accept three forms: a scalar, a one-dimensional list of length `C`, or a nested list of shape `(C, 1)`.
2. ★ The forward pass is **two lines**: first `y = self.m + torch.exp(self.logs) * x`, then `y = y * x_mask`. **The mask applies to the whole expression**, so a position whose mask is 0 outputs `0` (not `m`). This differs from the behaviour of coupling, see section 5.
3. `logdet = torch.sum(self.logs * x_mask, [1,2])` -- **the log-det is multiplied by the mask** (the opposite of coupling).
4. The reverse is `(x - self.m) * torch.exp(-self.logs) * x_mask`, with the mask also applying to the whole expression.

## 5. ★★ `ResidualCouplingLayer`: the mask multiplies only the x1 term

Official `modules.py`:

```python
class ResidualCouplingLayer(nn.Module):
  def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, mean_only=False):
    ...
    assert channels % 2 == 0
    ...

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

Five key points:

1. ★★ **The forward pass `x1 = m + x1 * torch.exp(logs) * x_mask`** -- the mask **multiplies only the `x1` term**, and the `m` term **is not masked**. So at a position whose mask is 0 the output is **`m`**, not `0`. This differs sharply from the "whole-expression mask" of `ElementwiseAffine` and is the single most important discriminator of this task.
2. ★★ **`logdet = torch.sum(logs, [1, 2])` is not multiplied by the mask.** Even where `x_mask` is 0, the `logs` at those positions still count toward the log-det. (Intuitively this looks odd -- but officially `stats = self.post(h) * x_mask` has already folded the mask into `logs` upstream, so "not multiplying by the mask" is correct there. In this task `logs` is taken as a directly supplied quantity, so it shows up as "the logs at masked positions still count toward the log-det".)
3. ★ `x0` is **passed through unchanged**; the official code neither masks nor transforms it.
4. With `mean_only=True` we get `logs = torch.zeros_like(m)`; with `mean_only=False` we get `m, logs = torch.split(stats, [half]*2, 1)`. The official `ResidualCouplingBlock` uses **`mean_only=True` everywhere**, so the block's log-det is always 0.
5. `assert channels % 2 == 0`; the reverse branch is `x1 = (x1 - m) * torch.exp(-logs) * x_mask` (mask on the whole expression) and it returns no log-det.

### ★ The official `x_mask` shape is not uniform

This is the easiest trap of the whole task:

| Class | Required mask shape |
| --- | --- |
| `Log` | same shape as `x`, `(B, C, T)` |
| `ElementwiseAffine` | same shape as `x`, `(B, C, T)` |
| `ResidualCouplingLayer` | `(B, 1, T)`, broadcast along channels |
| `ResidualCouplingBlock` | `(B, 1, T)` |

The `x_mask` that the official `ResidualCouplingBlock` receives comes from `commons.sequence_mask` and has shape `(B, 1, T)`. So when doing the cross-check you **must prepare two masks**: `(B, 1, T)` for coupling/block, and the expanded `(B, C, T)` for `Log` / `ElementwiseAffine`. This task accepts both shapes as input and normalizes them internally to `(B, C, T)`.

## 6. ★★ `ResidualCouplingBlock`: forward order / reverse order

Official `models.py`:

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

Four key points:

1. The flow sequence is fixed as `[Coupling_0, Flip, Coupling_1, Flip, ...]`, with the **coupling first and the flip second**.
2. ★ **Forward order forwards, reverse order reverses** (`reversed(self.flows)`). Coupling and flip **do not commute**, so swapping the order changes the result.
3. Every coupling uses `mean_only=True` → `logs` is always 0 → **the block's log-det is always 0**.
4. The official forward pass discards the log-det (`x, _ = flow(...)`), and the reverse branch even returns only `x`.

### ★★ The order difference is "accidentally unobservable" at n=2

This is an extremely subtle trap we hit while designing this task. Measured (`C=4, T=2`, mask all ones):

```
reverse order [F, C1, F, C0]  ->  [0.875, 1.25, 2.5, 4.25, 4.9, 5.8, 6.7, 7.6]
forward order [C0, F, C1, F]  ->  [0.875, 1.25, 2.5, 4.25, 4.9, 5.8, 6.7, 7.6]
```

**The two are identical!** The reason: with `mean_only=True` the coupling degenerates to "add a constant to the second half", and a flip is a channel reversal -- at `C=4` (two channels per half) the composition of these two steps happens to be symmetric with respect to "forward order vs reverse order".

At `n=1` (a single coupling) the difference becomes visible:

```
reverse order [F, C0]  ->  [3.0, 4.0, 0.5, 1.75]
forward order [C0, F]  ->  [2.5, 3.75, 1.0, 2.0]
```

⇒ **The judgement must use a single-coupling block to test the reverse order; two couplings cannot catch it.** This is the most valuable finding of this round's mutation iteration.

## 7. ★★ `StochasticDurationPredictor`: the sign of logq

Official `models.py`:

```python
logdet_tot_q = 0
e_q = torch.randn([x.shape[0], 2, x.shape[2]]).to(device=x.device, dtype=x.dtype) * x_mask
z_q = e_q
for flow in self.post_flows:
  z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
  logdet_tot_q += logdet_q

z_u, z1 = torch.split(z_q, [1, 1], 1)
u = torch.sigmoid(z_u) * x_mask
z0 = (w - u) * x_mask
logdet_tot_q += torch.sum((torch.logsigmoid(z_u) + torch.logsigmoid(-z_u)) * x_mask, [1, 2])
logq = torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q ** 2)) * x_mask, [1, 2]) - logdet_tot_q

logdet_tot = 0
z0, logdet = self.log_flow(z0, x_mask)
logdet_tot += logdet
z = torch.cat([z0, z1], 1)
for flow in flows:
  z, logdet = flow(z, x_mask, g=x, reverse=reverse)
  logdet_tot = logdet_tot + logdet
nll = torch.sum(0.5 * (math.log(2 * math.pi) + (z ** 2)) * x_mask, [1, 2]) - logdet_tot
return nll + logq
```

Six key points:

1. ★★ **The `lsig` term is subtracted.** `logdet_tot_q` is the sum of the `post_flows` log-dets plus `lsig`, while `logq = gauss - logdet_tot_q`. `lsig` is itself negative (`logsigmoid(z) + logsigmoid(-z) <= -log 4`), so subtracting it makes `logq` **larger**. Writing `gauss + lsig - post` is the most natural mistake, and it gets the sign completely backwards.
2. ★ `u = torch.sigmoid(z_u) * x_mask`, `z0 = (w - u) * x_mask` -- **`z0` is multiplied by the mask as a whole**.
3. `z0, logdet = self.log_flow(z0, x_mask)` -- `log_flow` is the `Log` primitive, and its `logdet` is `sum(-z0_log)` (the minus sign lives inside `Log`).
4. `z = torch.cat([z0, z1], 1)` -- `z0` (already passed through the log) and `z1` (not) are concatenated and the Gaussian NLL is computed over the concatenation.
5. The Gaussian terms of `nll` and `logq` have opposite signs: `nll` is `+0.5*(log(2pi)+z^2)`, `logq` is `-0.5*(log(2pi)+e_q^2)`.
6. `e_q = randn * x_mask` -- the noise itself is multiplied by the mask.

## 8. Three "reproduction pitfalls" of the official source

1. ★ **`modules.py` imports scipy but uses no scipy API at all.** The top of the official `modules.py` has `import scipy.signal`, but not a single scipy call appears anywhere in the file. To let the official module be imported directly for the cross-check, it is enough to drop an empty `scipy.py` stub into `/tmp/vits846/`.
2. ★ **`models.py` depends on the `attentions` module** (`from . import attententions`), which in turn needs things beyond `torch.nn.functional`. Importing the whole package fails. This task only needs the flow classes from `modules.py`, so the cross-check targets `modules.py` alone.
3. ★ **The `channels` of `ElementwiseAffine` is hard-coded at construction time.** If the cross-check script writes `ElementwiseAffine(1)`, it blows up with a shape mismatch on any case where `channels=2`. It has to be `ElementwiseAffine(len(m))`.

## 9. The three-layer structure of the judgement

| Layer | Role | Independence |
| --- | --- | --- |
| Official layer | directly imports `/tmp/vits846/modules.py`, uses a subclass to inject `m` / `logs` and bypass the WN network | the single source of truth |
| Reference layer | the `Fraction` implementation in `/tmp/vits846_solution_staged.py` | independently rewritten |
| Oracle layer | the inline `_oracle_*` functions inside the test cases | does not depend on the reference |

All three layers are cross-checked pairwise:

- Official vs reference: 22 cases, largest difference `8.882e-16` (the float64 rounding scale).
- Oracle vs reference: largest difference `1.776e-15`.

The oracle layer is the one that is actually asserted in the judgement, and it **does not import the reference**, so a wrong reference does not drag the oracle along with it.

## 10. ★ New lesson of this round: equivalent mutations must be pruned proactively

When the first round of mutation finished there were 5 survivors, of which **2 were truly equivalent mutations**:

1. `cp-split-half-wrong`: what I originally wrote inserted `if c == half: pass` into the `else` branch -- `pass` does nothing at all, so **the semantics did not change at all**.
2. `dur-z0-u-not-masked`: changing `u = _sigmoid(z_u) * mask` into `u = _sigmoid(z_u)`. It looks like "forgot to multiply by the mask", but `u` only ever appears inside `z0 = (w - u) * mask`, i.e. at a position that is multiplied by the mask as a whole -- when the mask is 0, `z0` is 0 no matter what `u` is. **Mathematically equivalent.**

Disposition: instead of hunting for test cases (which cannot be found), **replace them with different mutations**:

1. `cp-split-half-wrong` → `cp-split-halves-swapped`: change `if c < half:` into `if c >= half:` so the two halves are swapped (a real bug).
2. `dur-z0-u-not-masked` → `dur-u-sigmoid-removed`: replace `sigmoid(z_u)` with `z_u` (forgot the sigmoid, a real bug).

⇒ **★ Lesson: when a mutation survives, the first question is "is this a semantic gap or an equivalent mutation?"** Equivalent mutations must be **pruned and replaced proactively**, not left behind as "known survivors".

## 11. ★★ New lesson of this round: split the validation layer into mutually independent "defences"

There are three places where the mask takes part in the `logq` computation: `gauss`, `lsig`, `z0`. In the first round I only tested "the `z0` forgot the mask", so:

- `dur-gauss-mask-not-applied` (`gauss` forgot the mask) → survived
- `dur-lsig-mask-not-applied` (`lsig` forgot the mask) → survived
- `dur-z1-mask-not-applied` (the `z1` half forgot the mask) → survived

The reason: **those three places share the same mask layout in the test data**, and the test data's mask is 0 in exactly one spot ("batch 1, t=1") -- relying on that single spot alone leaves many mask-related bugs unobservable.

Disposition: split the three into **three mutually independent defences**, each with its own mutation and its own **observable assertion**:

```python
# three independent defences
gauss += -0.5 * (2.0 * half_log_2pi + float(eq[i]) ** 2) * float(mf[mi])
lsig  += (_log_sigmoid(zv) + _log_sigmoid(-zv)) * float(mf[mi])
acc   += 0.5 * (2.0 * half_log_2pi + zi1 ** 2) * float(mf[mi])
```

⇒ **★ Lesson: never validate the same semantics redundantly at several layers, or mutations will survive. Split it into independent cases where "each defence tests exactly one thing".**

## 12. ★★ New lesson of this round: unobservable differences need dedicated cases

The three survivors had three different root causes:

| Surviving mutation | Root cause | Disposition |
| --- | --- | --- |
| `log-reverse-exp-not-masked` | test gap: the reverse `Log` cases all used mask=1 data | add a reverse case with mask=0 |
| `cp-logdet-masked` | anchor defect: it multiplied by `mf[C-1][0]`, whose value happens to be constantly 1 | re-anchor it to the mask element of "the matching position" |
| `blk-order-not-reversed` | at n=2 the forward-order and reverse-order values coincide | add an n=1 single-coupling case |

The lesson of `cp-logdet-masked` is the most typical. My original mutation was:

```python
# wrong: the mask element multiplied happens to be constantly 1, equivalent to no change
acc += logslist[j] * mf[_index(b, C - 1, 0, B, C, T)]
```

A correct anchor must multiply by "**the mask element at the very position that this `logs` term corresponds to**":

```python
# right: locate the mask element by (b, t) for that logs term
acc += logslist[j] * mf[_index(b, 0, (j - b * half * T) % T, B, C, T)]
```

⇒ **★ Lesson: when writing a mutation, the anchor must land on "the input that can actually change".** A mask element that is constantly 1 turns the whole mutation into an equivalent one, and **it does not raise an error** -- it just silently survives.

## 13. Seven reusable lessons

1. ★★ In the `(B, C, T)` layout, `Flip` flips the **channel axis** (`torch.flip(x, [1])`) and **completely ignores the mask**.
2. ★★ In the forward pass of coupling the mask **multiplies only the x1 term** and `m` is not masked → at mask=0 the output is `m`; `ElementwiseAffine` instead uses a **whole-expression mask** → at mask=0 the output is `0`. The two must never be conflated.
3. ★★ The log-det of coupling is **not** multiplied by the mask, while the log-det of `ElementwiseAffine` **is**.
4. ★★ In `logq` the `logsigmoid` term is **subtracted** (it is itself negative).
5. ★ The minus sign of `logdet = sum(-y)` lives inside `Log`; `ElementwiseAffine` uses `sum(logs * mask)` (plus sign).
6. ★★ The official `x_mask` shape is **not uniform**: coupling/block want `(B, 1, T)`, `Log`/`ElementwiseAffine` want `(B, C, T)`.
7. ★★ The reverse of a block executes the flows in **reverse order**, but this difference is **unobservable at n=2** and must be tested with **n=1**.

## 14. The three-layer verification structure (final state of this round)

- Official cross-check: 22 + 2 cases, `official-vs-ref` at most `8.882e-16`, `oracle-vs-ref` at most `1.776e-15`.
- Mutation: **all 39 items CAUGHT, 0 SURVIVED** (four rounds of iteration: 28/36 → 36/36 → 39/39, including pruning 3 equivalent mutations and re-anchoring 3 of them).
- Test cases: 4 `check_*` functions (`check_flow_primitive` / `check_coupling_layer` / `check_coupling_block` / `check_duration_nll`), each carrying an inline oracle.
