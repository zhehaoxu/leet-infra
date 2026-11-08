# Notes on the official mechanism and its pitfalls

## Official source and version

- Repository: `NX-AI/xlstm`, default branch `main`.
- Files used: `xlstm/blocks/slstm/src/vanilla/slstm.py` (`slstm_forward_pointwise`), `xlstm/blocks/slstm/src/vanilla/lstm.py` (the plain LSTM contrast), `xlstm/blocks/slstm/cell.py` (`sLSTMCellConfig`, `reset_parameters`, `_zero_state`), `xlstm/blocks/slstm/src/cuda/slstm_pointwise.cu` and `slstm_forward.cu` (kernel semantics cross-check), `xlstm/components/init.py` (`bias_linspace_init_`).
- How it was fetched: the GitHub API, `git/trees/main?recursive=1` plus `contents?ref=main` plus `base64.b64decode`.
- The official cross-check script uses `.venv-eval/bin/python` (torch 2.14.1) and **execs the official `vanilla/slstm.py` as-is**, builds the `[B=1, 4H]` and `states[4, 1, H]` tensors with torch, and compares field by field against the hand-written plain-Python solve.

## Boundaries against mLSTM (problem 837)

Problem 837 judges the **mLSTM** in `xlstm/blocks/mlstm/backends.py`: a **matrix memory** `C` (DH×DH), the `sqrt(DH)` scaling, and the intra-chunk gates of `chunkwise_simple`. This task judges the **sLSTM** in `xlstm/blocks/slstm/src/vanilla/slstm.py`:

- the state dimensions differ: mLSTM has the matrix `C` plus the vectors `n` and the scalar `m`; sLSTM has a **single scalar per channel** `c` plus the scalars `n` and `m`, i.e. four states `[y, c, n, m]`;
- the gating differs: mLSTM stabilizes with a `logaddexp`-style `max` (`m_new = max(log_fg_act + m_state, igate_preact)`), while sLSTM uses `m + logsigmoid(fraw)` together with a hard `min(·, 1)` clip;
- the files differ and the judging domains do not overlap. This task does **not** judge the three-form equivalence, matrix-memory equivalence, or chunkwise behaviour.

## The five pitfalls of the sLSTM pointwise forward

1. **`torch.all(n == 0.0)` is a frame-level branch**. In the official vanilla implementation `n` is the whole `[B, H]` tensor and `torch.all` acts on that entire tensor, so either the whole frame takes `mnew = iraw`, or the whole frame takes `mnew = max(iraw, logfplusm)`. A **per-element** decision is wrong — only the CUDA kernel side is per-element (`eq_zero_g(n_cur)`). This is the sharpest contrast between sLSTM and the mLSTM stabilized gating, and the task statement follows the frame-level semantics of the vanilla implementation.
2. **`logfplusm = m + logsigmoid(fraw)`**. `fraw` first goes through `logsigmoid` (not `sigmoid`) and is then **added** to the previous frame's `m`. Dropping `m_prev` disables the stabilizer entirely.
3. **Both `igate` and `fgate` are hard-clipped by `min(·, 1)`**. `torch.minimum(torch.exp(...), torch.ones_like(...))` is an explicit upper bound, so even an `exp` result above 1 is pushed back to 1.
4. **`tanh` applies only to `zraw`**: `cnew = fgate * c + igate * torch.tanh(zraw)`. Writing `tanh(cnew)` or putting the `tanh` anywhere else gives a completely different result.
5. **`ynew = ogate * cnew / nnew`**, whose denominator is the **normalizer state** `nnew`, not `cnew`. sLSTM normalizes the scalar memory with `n`, which is the key difference from a plain LSTM.

## Pitfalls of the max stabilizer and the bias initialization

- `m_new = max(iraw, logfplusm)` guarantees `igate = exp(iraw - m_new) <= 1` and `fgate = exp(logfplusm - m_new) <= 1`, so both gates are naturally at most 1 and the `min(·, 1)` clip only really bites for `fgate` on the `zero_n_branch` branch (where `m_new = iraw`, so `fgate` can be far above 1). This matters when writing mutations: the `igate` clip is a **never-firing** equivalent.
- The `powerlaw_blockdependent` initialization uses the exponent `0.3 + 1.3 * ratio` with `ratio = _block_idx / (_num_blocks - 1)` (and `0.0` when `_num_blocks == 1`); the base is `arange(head_dim) / (head_dim - 1)`, and **the minus sign sits outside the parentheses** (`-( -5 + 12 * base^exp )`), so a base of 0 gives `+5`.
- When `head_dim == 1`, `arange(1)/(1-1)` is `0/0`, so the bias is `NaN` in `powerlaw` mode; in `small_init` mode `torch.linspace(3.0, 6.0, 1)` gives `[3.0]`.
- The `small_init` bias of the `f` gate is `torch.linspace(3.0, 6.0, head_dim)`, **endpoints included**; every other gate's bias is zero.

## How it was verified and how the test cases were designed

- Official cross-check: `pointwise_step` 600 groups, `max_stabilizer` 600 groups, and every `head_dim × num_blocks × block_idx × mode` combination of `bias_powerlaw` — **all consistent**.
- Test cases: three `check_*`, each with an independent plain-Python oracle, hand-computed anchors and a grid comparison.
- The hand-computed anchors pin down: the frame-level `torch.all` branch, the `m_prev` inside `logfplusm`, the hard `min(·,1)` upper bound, `tanh` applying only to `zraw`, the denominator of `ynew` being `nnew`, `y_new` being `NaN` when `n_new == 0`, the powerlaw exponent and the minus sign outside the parentheses, the `NaN` at `head_dim == 1`, and `linspace(3.0, 6.0)` in `small_init`.
- Mutation testing: 42 single-point mutants, all CAUGHT, with the **double-mode self-check of the correct solution confirmed ACCEPTED beforehand**.

## Pitfalls hit during cross-checking and mutation (reusable lessons)

1. **The comparison function of the cross-check script must support nested lists**. The first version of `cmp_value` only handled scalars, and the 4×H structure of `raw_gates` immediately raised `TypeError: float() argument must be ... not 'list'`.
2. **The expected values must use `torch.nn.functional.logsigmoid`, never `torch.log(torch.sigmoid(x))`**. The latter applies sigmoid and then log, and near x = 0 its cancellation error is about 2e-8, one order of magnitude away from the official implementation. The official source uses `from torch.nn.functional import logsigmoid`.
3. **`torch.tensor(x)` in the cross-check defaults to float32**, so the exponential operations carry a relative error around 1e-7. `dtype=torch.float64` must be given explicitly.
4. **Distinguish "truly equivalent" mutations from coverage gaps**. The first round had 5 survivors out of 41 mutants, and diagnosis showed all of them were truly equivalent:
   - swapping `wx` and `ry` in `raw = wx + ry + b` is the commutativity of addition;
   - the `min(·,1)` on `igate` never fires (`m_new >= iraw` always holds);
   - `gate_le_one` is always True (both gates are clipped to `<= 1`);
   - `_ = y_prev` is a no-op.
   Per the established principle, truly equivalent mutants must be dropped and replaced with testable forms (respectively "flip the sign of the bias", "per-element branch", "strictly less than", "`m_prev` reads the wrong state").
5. **A mutation with too narrow a window is effectively equivalent.** Changing `if gf > 1.0` into `if gf > 1.0000001` is almost never hit by the grid and still survived. Changing it into "the clipped value becomes 0.999999" is finally testable.
6. **Hand-computed anchors must be re-checked by hand.** This round the `max_stabilizer` anchor 1 missed the `min(·,1)` clip on `fgate` (channel 1's raw value 4.154 should be 1.0), and the `pointwise_step` anchor 3 miscomputed `c_new = fgate*c + igate*tanh(zraw)` as 1.5 (the correct value is 0.5). The self-check caught both immediately.

## Boundaries kept out of the judging domain

The GEMM between the `R` matrix and the state together with its `reshape`/`transpose` layout chain, the CUDA kernel launch configuration and `round_to_power2`, `SLSTM_FORWARD_CLIPVAL`, autograd backward passes and gradient clipping, the random initialization of parameters (the `zeros`/`standard` weights, the normal sampling of `small_init_init_`), the plain LSTM variant in `vanilla/lstm.py`, multi-head splitting and configurations with `num_gates != 4` are all kept out of the judging domain. The task keeps only the three deterministic pure numeric ledgers.
