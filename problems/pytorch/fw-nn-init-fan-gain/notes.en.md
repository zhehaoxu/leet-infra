# notes — fw-nn-init-fan-gain (order 710)

## Provenance

- Source: the judge environment's own `.venv-eval` `torch/nn/init.py` (v2.14.1, installed 2026-10-19) — more authoritative than a GitHub fetch since judging runs this exact version; `calculate_gain` / `_calculate_fan_in_and_fan_out` / `_calculate_correct_fan` / `kaiming_uniform_` / `kaiming_normal_` / `xavier_uniform_` / `xavier_normal_` verified line by line. He init paper 1502.01852, Glorot 2010.
- Scope vs the todos candidate: only the fan/gain/std/bound formula-chain ledger; `trunc_normal_` (erf/CDF reparametrization), `orthogonal_` (QR), `sparse_`, `dirac_` excluded.

## Judge surface

- solve(noises, probe=None): explicit-noise-input paradigm (no global RNG dependency — torch.Generator only appears in tests to build the fixed noise pack); probe=[nonlinearity, param] returns `{"probe_gain": ...}` only, exposing calculate_gain's validation paths (bool param, unknown nonlinearity) which are unreachable from the fixed scenarios.
- Oracle: `test_cases._py` calls the official torch.nn.init functions directly (`tinit.calculate_gain` / `tinit._calculate_fan_in_and_fan_out` / `tinit._calculate_correct_fan`) as the hardest anchor, with the value mapping chained from the docstring formulas (bound = gain·√(3/fan), xavier bound = gain·√(6/(fi+fo)) — deliberately different from the solution's official √3·std implementation path, with the tolerance absorbing ULP). `check_vs_torch_fill` is end-to-end: the solution's bound is checked against the min/max extremes of a large official `kaiming_uniform_` sample ([256,3,5,5], fan_in=75).
- Float comparison: the noise pack is float64 (float32 ULP ~6e-8 would break the 1e-12 tolerance); scalars use isclose(rel 1e-12), values use torch.allclose(rtol 1e-12).

## Anchors

- fan: [3,5]→(5,3), [4,3,2,2]→(12,16), [2,4,3]→(12,6) (receptive field = product of shape[2:]).
- gain: bool param rejected (official condition `not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float)` parses as `(A and B) or C`, so bools hit the else raise); int 0 legal (slope=0); leaky default 0.01.
- **kaiming's `a` defaults to 0 ≠ calculate_gain's slope default 0.01**: the all-default scenario has gain=√2 (misusing 0.01 gives √(2/1.0001), caught by the bound case).
- xavier: fi+fo denominator ([4,2,2,2] → fi=8, fo=16, sum 24); bound=√3·std.
- uniform_(a,b) value mapping: `a + (b-a)·(noise+1)/2` (noise∈[-1,1] → U(a,b)); param pinned to symmetric half-width (kaiming/xavier = bound, plain = (b-a)/2).

## Mutation log (13/13 caught)

fan-receptive-drop, fan-swap, gain-tanh (5/3→1.5), gain-slope-dflt (0.01→0), gain-bool-allow, gain-unknown-ok, kaiming-a-default (a=0→0.01), kaiming-bound-sqrt3, xavier-bound-sqrt3, xavier-fi-fo-max, mode-lower (uppercase FAN_IN scenario), uniform-map-formula, kaiming-normal-std-neg.

Two equivalent mutations recorded: xavier bound official path √3·std vs direct gain·√(6/(fi+fo)) (ULP absorbed); receptive-field multiplication order reversed (integer commutativity).

## Red lines

No torch.nn.init calls (the solution hand-writes the whole chain); no trunc_normal_/orthogonal_/sparse_/dirac_; no global RNG dependency (noise explicit); boundary with `fw-distributed-sampler` (253): this is the initialization formula-chain ledger, not sampler assembly.
