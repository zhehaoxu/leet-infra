# Solution notes — add_weighted_adapter

## Official account

- HF PEFT `peft/tuners/lora/model.py` (LoraModel.add_weighted_adapter +
  _check_add_weighted_adapter + _generalized_task_arithmetic_weighted_adapter +
  _svd_generalized_task_arithmetic_weighted_adapter); offline weighted adapter
  merging.
- Judging domain: rank rules, scaling normalization, the cat/linear/svd merge
  paths, bias merging. Embedding/Conv branches and the
  TIES/DARE/ties_svd/dare families are out of scope (`peft-ties-dare-merge`
  covers the paper-level operators); the modules_to_save/target_parameters/
  DoRA/KaSA preflight ValueError branches are not part of the problem.

## Pitfalls

- **A placeholder anchor value is a live FAIL truth** (replayed for the N-th
  time in this tree): all four anchors were backfilled from actual solution
  runs; guessed values always fail.
- **The clamp test's quantile can be negative**: when the q=0.5 median lands in
  the negative range, `clamp(-hi, hi)` inverts its bounds (min > max) and torch
  truncates every element to max — the mutant and the expectation are then
  **wrong in the same way**, a false mutation pass. Use q=0.9 and assert
  `hi > 0` explicitly.
- **sqrt discrimination in the linear branch**: if every `|w·s|` in the test
  data equals 1 (e.g. 0.5×2, −2×0.5), the sqrt-removal mutant is indistinguishable
  (√1 = |1|) — the weight/scaling combination must produce `|w·s| ≠ 1`
  (0.5×2 = 1 and −2×1 = 2) to kill the `sqrt→abs` mutant.
- **The svd S is absorbed into B**: `U@diag(S)` and "multiply S into Vh" have the
  same reconstruction product (associativity), so reconstruction assertions
  cannot kill that mutant — pin the kohya convention via `B`'s column norms ==
  |S| (U's columns are orthonormal).
- **Vh truncation slicing out of range is silent**: `Vh[:rank+1, :]` returns the
  original tensor without error when rank == min_dim — the truncation assertion
  must run on a case where rank is strictly below min_dim (new_rank = 2 < 6).
- **Assert the single-source-forces-linear return value**: not just new_rank —
  the combination_type field is in the returned dict and must be asserted too
  (otherwise the `cat→linear` rewrite leaks: a single-source cat has the same
  new_rank as the linear rule).
- The svd branch combines with the **raw weights** (scaling is already inside the
  delta); only linear/cat multiply the scaling explicitly — this three-way
  inconsistency is the problem's biggest exam point.

## Mutant list (15)

resolve: single-source force→cat, linear equal-rank check removed, cat sum→max,
svd max→min, rank_pattern max→min, scaling ÷→×; cat: B wrongly scaled, B concat
dim=1→0 (shape error); linear: sign dropped on the A side, sqrt→abs on the B
side; svd: delta scaling dropped, S absorption into B dropped, Vh truncation +1,
clamp symmetric→one-sided; bias: scaling dropped. 15/15 CAUGHT.
