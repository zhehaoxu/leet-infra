# nn.init fan/gain/std/bound Ledger

## Problem

The six initializers in `torch.nn.init` (Kaiming uniform/normal, Xavier uniform/normal, `uniform_`, `normal_`) all reduce to one formula chain: **fan computation → gain table → std/bound → noise scaling**. Hand-write the entire chain (noise supplied explicitly, no `torch.nn.init` calls): given a fixed noise pack `noises`, emit the gain table, fan table, and `dist/param/values` ledgers for eight scenarios.

## Solution

Official semantics (`torch/nn/init.py` v2.14.1):

- **fan**: `fan_in = size(1)·prod(shape[2:])`, `fan_out = size(0)·prod(shape[2:])`; `dim<2` raises `ValueError`. `mode.lower()` must be in `{fan_in, fan_out}` (uppercase legal);
- **gain**: linear/conv{1,2,3}d/conv_transpose{1,2,3}d/sigmoid → 1; tanh → 5/3; relu → √2; selu → 3/4; leaky_relu → √(2/(1+slope²)) with slope default **0.01**; param must be a non-bool int or float (bool subclasses int; officially rejected) — otherwise `ValueError`;
- **kaiming**: `gain = calculate_gain(nonlinearity, a)` — note `a` is the negative slope with default **0** (not calculate_gain's own 0.01 default); `std = gain/√fan`; uniform `bound = √3·std`;
- **xavier**: `std = gain·√(2/(fan_in+fan_out))`; uniform `bound = √3·std` (= `gain·√(6/(fi+fo))`; the two paths are equivalent within tolerance);
- **value mapping** (noise explicit): kaiming/xavier uniform values = `bound·noise`, normal = `std·noise`; `uniform_(a,b)` = `a + (b-a)·(noise+1)/2`; `normal_(mean,std)` = `mean + std·noise`.

Scenarios include: conv relu fan_in (mode given as uppercase `FAN_IN`), leaky slope=0.2 fan_out, all-default kaiming (a=0 → gain=√2), kaiming_normal relu fan_out, xavier gain=5/3, all-default xavier, 1D `uniform_` (no fan), and `normal_(1,2)`; a probe channel unit-tests calculate_gain's validation paths.

## Constraints

- The noise pack has 8 keys matching the scenarios; shape mismatch or missing key → `ValueError`; non-dict `noises` → `ValueError`.
- probe is `[nonlinearity, param]`: unknown nonlinearity, bool/str param → `ValueError`.
