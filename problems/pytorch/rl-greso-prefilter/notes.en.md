# Implementation notes (788 rl-greso-prefilter)

## The official ledger, line by line

### presampling_selector.py::DataProfiler (skip family)

- `add_reward`: `data[f"{source}_{index}"].append((epoch, reward))`.
- `easy_example_skip(current_epoch, data_id, n=-1)`: any `(epoch, reward==1)` in the trace with `epoch + n ≥ current`; n=−1 → 10e7 (forever).
- `easy_linear_backoff_skip(epoch, data_id, k)`: trailing run of `reward ≥ 0.98` (breaks below 0.98) → count; count==0 early return; `backoff = randint(0, k·count)`; `max_epoch + backoff ≥ current` → True. max_epoch spans the **whole trace**.
- `hard_linear_backoff_skip`: gate — the **latest** reward >0.11 → False (solvable examples are never hard-skipped); trailing run `≤ 0.11` count (breaks above 0.11); otherwise as easy.
- `easy/hard_probabilistic_skip(p)`: same counts (no count==0 early return); `skip_prob = 1 − max(p^count, 0.01)`; `torch.rand(1) < skip_prob`. **Lazy rand** — short-circuited gates never sample.
- `filter_examples_all_probabilistic(epoch, batch, p_easy, p_hard, return_log)`: easy first then hard; skip_log `[data_id, 0|1|2]` (0=easy, 1=hard, 2=kept).

### ttis.py::filter_zero_advantage

- reward reshaped (−1, n); `max(dim=1) == min(dim=1)` → zero-variance group; kept groups expand as `idx·n + arange(n)`.

### ray_trainer.py tail (scheduling ledger)

- `targeted_easy = target_zero_variance/3`, `targeted_hard = ×2/3` (the hard side keeps 2/3 exploration headroom).
- Filtered accumulation: `acc_batch = concat(acc_batch, filtered)`; released only when `len(acc_batch) ≥ adaptive_sampling_bs` (sliced to an n_gpus multiple), otherwise continue.
- Refill: `estimated = int(reminder / (1 − zvr) / n * 1.25)`; `estimated ≤ default → last_sampling=True`; `adaptive = clamp(estimated, 64, default)`; `full_sampling_bs` → adaptive=default, last=False.
- p adaptation (epoch > 0): `hard_ratio ≥ targeted_hard → p_hard = max(p−0.01, min_p)` else `min(p+0.01, max_p)`; easy likewise against targeted_easy.

## Sandbox engineering

- /tmp/greso_check.py (python3, stdlib only): the DataProfiler class transcribed verbatim with randint/rand replaced by deterministic stub sequences (`lambda a, b: b − 1` etc.).
- **Lesson: lazy rand semantics** — the first solution popped rands eagerly in the filter layer, contradicting the official "short-circuited gates never sample" behavior → check_filter IndexError/misalignment. Parameterized random sources must be callables (`rand_fn()` invoked only when the random comparison is actually reached).
- json serialization turns dumped tuples into lists — tests normalize with `[list(t) for t in ...]`.

## Judging protocol

The judge injects a single `solve`; `mode` kwarg dispatches seven entries. `check_*(solve)` assertion set, 9 cases (stdlib only, no torch dependency — random sources fully parameterized).
