# Active-Prompt Uncertainty-Based Sample Selection

## Problem

Human annotation for CoT demos is expensive. Active-Prompt (arXiv:2302.12246) uses **uncertainty** to pick the questions worth annotating: sample the model k times on the same question — questions with wildly varying answers mean the model has not learned them yet and go to humans first; highly consistent questions are skipped. This is active learning's query-uncertainty transplanted into prompt engineering.

You are given (GPT sampling and answer extraction happen outside the pool; this problem has no clustering): `samples_per_question` — for each question, k **post-extraction** answer strings (an empty string means that sample had no answer); plus three knobs `sort_by` / `dataset_kind` / `qes_limit`.

## The ledger to implement

### 1. occurrence counting

Per question, per sample: non-empty answers are counted under their **raw string** as key (dict in first-seen order); empty answers go to the sentinel key `NO_SOLUTION = "-10086"`. `strategyqa` occurrences **preset** `{"yes": 0, "no": 0}`.

### 2. variance (numerical only)

```
ans_list = [float(ans) for ans, c in occurrence.items() for _ in range(c)]
variance = population variance (ddof=0)
```

**NO_SOLUTION also enters the variance as -10086** — official as-is; a question with no successful samples gets a huge variance.

### 3. entropy (scipy.stats.entropy convention)

```
entropy = -Σ_{c>0} (c/n)·ln(c/n)      # zero-frequency terms contribute 0
```

All-zero frequencies (n=0) yield nan. No scipy in the judging env; the self-implementation matches: `entropy([2,2]) = ln2`, a single-value distribution is 0.

### 4. disagreement

`len(occurrence)` — the number of distinct answers (including the NO_SOLUTION key; strategyqa's preset yes/no always count as two).

### 5. Sorting (stable, ties keep input order)

| sort_by | key | notes |
|---|---|---|
| disagreement | `-len(occurrence)` descending | more distinct answers first |
| disagreement + strategyqa | `abs(yes - no)` **ascending** | closer votes = more uncertain; the official except fallback is a dead branch |
| variance | `-variance` descending | **numerical only**; the official elif filters by dataset and silently skips sorting otherwise |
| entropy | `-entropy` descending | all tasks |

### 6. qes_limit

Compute only the **first qes_limit questions** (0 = all; beyond the count means all); sorting applies only to the computed subset.

Return the records list (sorted), each `dict(dataset_idx, variance?, entropy, occurrence, disagreement)` with `dataset_idx` the original input index.

## Input constraints

- `samples_per_question` is a non-empty list[list[str]] with non-empty per-question lists (num_trails ≥ 1);
- `sort_by` ∈ {disagreement, variance, entropy}; `dataset_kind` ∈ {generic, numerical, strategyqa};
- `qes_limit` is an integer ≥ 0 (bools do not count).

## Hints

- python's `sort(key=..., reverse=True)` keeps ties in **original order** (it does not reverse them), identical to a `-key` descending sort;
- `qes_limit == n` caps or not with the same result;
- A zero-frequency entry (e.g. the preset no:0 when every strategyqa vote is yes) contributes nothing to the entropy but **does count toward disagreement**.
