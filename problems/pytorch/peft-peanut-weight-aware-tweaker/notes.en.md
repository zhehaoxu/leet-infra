# Notes

- Source: PEANuT (arXiv:2410.01870), merged into HF PEFT `src/peft/tuners/peanut/` (`PeanutConfig`: r=32, depth=0, act_fn='relu', scaling=1.0, init_weights=True).
- Judging domain: the pure-function core of `_compute_delta_weight` + the parameter-count arithmetic + initialization RNG replay + residual-pairing assertions; the official layer class's device/dtype shuffling and merge caching are out of scope.
- Modes (6): delta_weight / forward / param_count / init_factors / zero_init_check / residual_order.
- Anchors in float64, tolerance 1e-9; `kaiming_uniform_(a=√5)` equals `uniform(±1/√fan_in)`, replayed directly via `uniform_(generator=)` to avoid version differences in `nn.init` generator support.
- The reverse-order residual is distinguished end-to-end with identity activation and diagonal `i+1` encoder/decoder weights: different orders necessarily give different results.
- mut 17: residual reverse/pop / no-residual / scaling / transpose / B transpose / three activation sites / enc-dec skip-first / lora count / depth count / RNG order / bound / zero-init / forward direction / zero-check flip — all CAUGHT.
