# Reference Approach: Loop over Steps, Vectorize over Experts, Keep an Explicit State Machine

## Implementation

1. Write the EMA recursion literally: `keep = beta`, `rise = 1 - beta`, `ema` initialised to `zeros(E)`, then **advance it on every step** with `ema = keep * ema + rise * load[step]`, the first step included — that is where `ema[0] == (1-beta)*load[0]` comes from.
2. Inside a step compute `alive = ~is_dead`, `below = ema < death`, `above = ema >= revive` first, then advance only the counter belonging to the current state: `miss = where(alive & below, miss+1, where(alive, 0, miss))`, with `back` symmetric to it.
3. `dies = alive & (miss >= patience)`, `wakes = is_dead & (back >= revive_patience)`. Because `alive`/`is_dead` are complementary the two masks are mutually exclusive, so they can be applied together in one `where` chain.
4. On a flip, clear `miss` **and** `back` together, then update `is_dead`, and finally write `dead[step]` plus `streak[step] = where(is_dead, back, miss)`.
5. Accumulate the three ledgers in the same loop: `first_dead = where(dies & (first_dead < 0), step, first_dead)` (the `< 0` latch keeps only the first death), `changes += dies + wakes`, `dead_steps += is_dead`.

Four common mistakes:

- **Writing the comparisons as `<=` / `>`.** Death uses strict less-than, revival uses greater-or-equal; this is the core convention of the statement. Flipping one sign turns the hysteresis band into a closed interval and shifts the behaviour on every boundary value.
- **Judging state with the *previous* step's EMA**, or putting the state machine before the EMA update. Being off by one step looks like it only moves one flip, but `first_dead`, `state_changes` and `dead_steps` all cascade from it.
- **Recording `dead[s]` as the pre-flip state.** `dead[s]` means "after this step is applied", so the step that kills an expert must already read True.
- **Forgetting the `< 0` latch on `first_dead`**, so every death overwrites it and the ledger ends up holding the last death instead of the first.

## Correctness checks

Eight `check_*` functions: the hand-computed example (all six return values verified), random differential testing (8 combinations of `beta`/thresholds/patience/load spread, plus three non-contiguous views), state-machine invariants (`dead_steps == dead.sum(0)`; `first_dead` agrees with the first True in each `dead` column; `state_changes` equals the flip count recounted from the `dead` columns including step 0; the flipping step has `streak == 0`; adjacent steps in the same state have `streak ∈ {0, previous+1}`; the active counter never reaches its own threshold; the EMA recursion re-checked bit-for-bit against the cold-start convention), hysteresis band (while the EMA sits inside `[death, revive)` an alive expert gains no miss and a dead expert gains no back), threshold boundaries (`death == revive` with the EMA exactly on the threshold: the alive side does not count a miss, the dead side does count a revival; with `beta = 0`, `ema == load`), patience and cold start (`patience > S` never kills, `patience = 1` kills on the very first step, single-step history, all-zero load), invalid inputs, and the maximum shape `32×16`.

The oracle is pure Python: per expert, per step, it walks the same state machine with scalar `float`/`int`, borrowing no tensor operators and not reusing the mask formulation of the reference solution.

Mutation testing caught 10 mutants: `<=` for death, `>` for revival, the off-by-one `miss > patience`, a one-step-lagged EMA, EMA seeded with `load[0]`, `dead[s]` recorded pre-flip, `streak` always emitting `miss`, `first_dead` overwritten on every death, `state_changes` counting only deaths, `dead_steps` counting the pre-flip state. Four further mutants are **indistinguishable, because the statement makes them equivalent**: the statement requires `death_threshold <= revive_threshold`, so "below the death line" and "not below the revive line" are mutually exclusive within one step, which means "advance both counters every step" and "clear only your own counter on a flip" cannot produce a different table; `torch.lerp(load, ema, beta)` is likewise indistinguishable from the explicit recursion under the `2e-11` tolerance. This is not a hole left in the judge — the inequality itself is what makes the state machine safe, and the statement enforces it as a hard validation so that changing the convention fails immediately.

## Scope and sources

"Track expert load with an EMA and retire an expert after enough consecutive steps below a threshold" is a common practice in sparse-MoE training, but the transformers 5.x used by this repository's judge environment contains **no** corresponding implementation (a whole-package search for `ema_load`, `dead_expert`, `jitter` only hits unrelated words), and no verifiable primary formula could be retrieved. Every convention in this problem is therefore **defined by the problem statement itself**, with no attribution claimed: EMA starting at 0, advancing every step, strict `<` for death, `>=` for revival, clearing counters on a flip, the two meanings of `streak`, and the definitions of the three ledgers. The dual-threshold hysteresis borrows the generic control-engineering notion of a Schmitt trigger, not a specific paper's implementation.

Division of labour inside the collection: single-step dispatch and dropping live in `moe-capacity-dispatch` and `moe-capacity-jitter-dispatch`; this problem only owns the **cross-step load ledger** — it neither dispatches nor rescales weights. Feeding load information back into routing biases is `moe-aux-free-routing` (DeepSeek-V3's bias update); the family of load-balancing losses is covered by `moe-switch-aux-loss`, `moe-cv-aux-loss` and `moe-sequence-aux-loss`.
