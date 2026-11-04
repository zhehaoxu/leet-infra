# Solution Notes

## Approach

The deterministic core of the official `mask_rebuild` (lib/opt.py L281-388), expanded along four orthogonal knobs:

1. **metric choice** (WD/W/D) — how importance is computed;
2. **find_threshold** — pairing the growth/prune sets to derive dual thresholds;
3. **granularity** (block/layer/output1/input1) — the threshold scope;
4. **retain_reconstruction** — how the combined weight is built.

The judging scope is the whole `mask_rebuild`; upstream dependencies (the grad² Fisher running average from block forward MSE backprop in `OPTAttentionGPT/OPTMLPGPT.add_batch`, and SparseGPT pruning that produced the initial sparse weights) are explicit inputs per discipline. The official N:M structured branch (per-row thresholds plus interleaved `ii % prune_m == 0` topk) is not implemented and is noted as a boundary.

## Key implementation points

- **The four branches of find_threshold** (faithful to the official code):
  - empty set → `(inf, -inf)`; neither strict comparison fires → mask unchanged (observable: thresholds contain inf/-inf, and rebuilt is all-zero or all-kept after mask zeroing);
  - `idx = int(recons_num * threshold)`, with `recons_num` counted using **strict >** (an exact 0 difference does not count);
  - the `idx == numel` guard takes the last element — the two sides have different numel (growth may be shorter than prune) and are checked independently;
  - pairing is **position-aligned** (growth descending vs prune ascending), not a same-rank value comparison — craft test data with `gs[i] - ps[i]` in mind, not the naive "i-th largest vs i-th smallest equal values".

- **Exchange order**: `mask[growth]=False` first, `mask[prune]=True` second. Since growth_indices uses metric > gt and prune_indices uses metric < pt, with gt taken from the high end of growth and pt from the low end of prune, the two sets rarely intersect; the order still follows the official code.

- **The action path of retain_reconstruction**: when alive positions satisfy ref==weights (an early fixture pitfall), retain is a no-op; only when alive positions have ref ≠ weights (SparseGPT-reconstructed weights) does combined diverge. rebuilt only zeroes via the mask and takes combined at alive positions — so retain=True and False may yield identical rebuilt; the divergence lives in **the pruned positions' metric** (hence thresholds and masks).

- **Internal precision**: the metric is float64 throughout. `.double()` on the combined side is redundant (abs/where introduce no rounding and float32→float64 is exact), while the scaler side hosts the only rounding point (sqrt) — proven by mutation testing: dropping only the combined side is equivalent (excluded), dropping only the scaler side is caught (float32 sqrt rounding leaks into the float64 product).

- **Returned thresholds**: `thresholds` maps name → `(gt, pt)`; output1/input1 return per-row/per-column arrays — tests use `np.allclose` for array thresholds and `==` for scalars.

## Test design (12 cases)

- Full oracle comparison: layer/block/output1/input1 × both retain states × three metrics, with an independent numpy oracle recomputing find_threshold and the bookkeeping;
- Observable exchange: at least one of revived (ref==0 and rebuilt≠0) or newly_pruned (ref≠0 and rebuilt==0) fires; masked positions of rebuilt are 0;
- Empty sides: fully pruned / fully alive matrix → inf/-inf → mask unchanged (rebuilt all-zero / all-kept);
- Strict > pairing: growth=[3,3], prune=[5,4,3] with an exact 0 first difference → recons=0 (a >= mutant yields recons=1 and pt 3→4);
- Out-of-range guard: growth of three 9s all greater than prune of two 1s → idx=2==prune count → pt takes the last element 1.0 (dropping the guard → IndexError);
- threshold=0/0.5/1.0 gives monotonically non-decreasing swap counts; equality boundaries (metric==gt does not revive, metric==pt does not prune);
- Purity: tensors in all three dicts are unmodified and the mask is a fresh tensor;
- Validation: empty dicts / key types / key agreement / shapes / negative scaler / metric / granularity / threshold (bool, 1.5, -0.1, inf, str rejected; int 0 accepted).

## Mutation closure

All 52 mutations caught + 1 equivalent excluded (m-comb-f32: dropping `.double()` on the combined side keeps exact numerics since abs/where do not round and the scaler-side double broadcasts; the only rounding point, sqrt, stays in double). Every branch has at least one dedicated case: the three metrics differ from each other (non-constant scaler), the four routing targets differ (block thresholds ≠ layer thresholds), m-order (exchange order) is caught via overlapping-position assertions, and message mutants via substring assertions.

## Engineering remarks

- The first fixture used `ref = w * keep`, making ref==w at alive positions and retain a no-op — a mocked reconstructed weight must differ from the dense value at alive positions (multiplied by 0.9).
- An early numpy oracle wrote the updated mask only to a local variable without storing it back into the dict, causing a false oracle mismatch — the oracle is a second implementation, so suspect the oracle first when debugging.
- The scaler factor is irrational (π) to raise the chance of float32/float64 rounding divergence, serving the float32-precision mutation.
