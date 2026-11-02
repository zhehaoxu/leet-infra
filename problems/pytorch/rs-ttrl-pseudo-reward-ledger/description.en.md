# TTRL Majority-Vote Pseudo-Reward Ledger

TTRL (arXiv 2504.16084) votes with the model's own samples **at test time**: draw `n`
completions for one question, pull the `\boxed{}` answer out of each, and the most popular
one becomes that question's **pseudo-label**. That pseudo-label then plays ground truth so
every sample can be scored, and the scores fold into a set of evaluation metrics.

This problem keeps only the **deterministic pipeline** (extract → canonicalize → vote →
reward → metrics → downsample): LLM generation, `sympy` simplification and mathd grading are
all turned into explicit inputs, mirroring `PRIME-RL/TTRL`'s
`verl/verl/trainer/ppo/ttrl_utils.py` and `verl/verl/utils/reward_score/ttrl_math/`.

Implement `solve(outputs, n, original_gt, canon=None, keep=None)`.

## Inputs

- `outputs`: a flat list of strings whose length must be a multiple of `n`; block `i`
  (indices `[i*n, (i+1)*n)`) holds the `n` samples of question `i`.
- `n`: samples per question, must be `>= 1`.
- `original_gt`: one ground-truth string per question; the count must equal the number of
  questions.
- `canon`: the simplification table `{raw answer: simplified}`, `None` means identity; a miss
  keeps the value verbatim.
- `keep`: optional downsampling width, see below.

## Answer extraction

Every sample is mined for an answer first:

- No `\boxed` substring → nothing extracted (`None`). `\fbox` does **not** count.
- Otherwise scan braces starting from the **last** occurrence of `\boxed`: `{` adds one,
  `}` subtracts one, and the **first** `}` that brings the depth back to 0 closes the box;
  never closing before the end of the string → `None`.
- The sliced candidate must start with `\boxed{`, else `None` — so `\boxed` followed by a
  space, or a brace pair that only balances further down the string, yields nothing, and an
  earlier well-formed box does not rescue it.
- On success the content between `\boxed{` and the closing `}` is returned: nested braces
  survive verbatim (`\boxed{\frac{1}{2}}` → `\frac{1}{2}`) and `\boxed{}` yields the
  **empty string**, which is a legal answer.
- Only `None` means failure; the empty string is a real vote.

## Pseudo-label and tally

- Samples that failed extraction are dropped; the rest each go through the table once —
  every raw value is canonicalized **exactly once**, and every later comparison is a plain
  string comparison between canonical values.
- Nothing left at all: the label is the **string** `"None"`, the ratio is `0.0`, and the
  tally is empty (it still takes part in grading, so a ground truth of `"None"` gives
  `label_accuracy` 1.0).
- Otherwise record `[answer, count]` in **first-seen order**; the highest count wins and
  ties go to the earliest occurrence.
- `ratio` = winning count / **`n`**: the denominator is the block size, not the number of
  successfully extracted samples.

## Rewards

- Pseudo-label reward: a sample that failed extraction scores 0; otherwise it is 1 when its
  canonical answer equals the pseudo-label.
- Ground-truth reward: when the ground truth itself contains `\boxed` it is extracted first
  and then canonicalized; a sample scores 1 when its canonical answer equals it.
- If the ground truth cannot be extracted (say `\boxed{5` is missing its closing brace) every
  ground-truth reward in that question is 0, and not even an empty answer matches.
- The pseudo-label is already canonical, so it never goes through the table again: a chained
  table (`a->b, b->c`) does not chain.

## Metrics

- Five numbers per question:
  - `label_accuracy`: does the pseudo-label equal the canonical ground truth (0.0 when the
    ground truth cannot be extracted);
  - `reward_accuracy`: the fraction of samples whose pseudo-label reward equals their
    ground-truth reward **sample by sample**;
  - `majority_voting_reward`: mean pseudo-label reward;
  - `ground_truth_reward`: mean ground-truth reward;
  - `pass@{n}`: 1.0 when the ground-truth rewards sum to `>= 1`, else 0.0.
- Batches average across questions and **take their key order from the first question**; the
  `k` in `pass@k` is the per-question sample count actually in effect.
- One extra key `majority_ratio` (mean of the per-question ratios) is appended at the end.

## Downsampling

- With `keep`, indices are taken first: `[i*n + t for i in 0..questions for t in 0..keep)`,
  and `selected` is exactly that list.
- The upstream code does **not** check `keep <= n`: with `keep > n` the indices bleed into
  the next block and the last block runs past the end → `IndexError`.
- The whole pipeline then reruns with `keep` as the new `n`: tally, ratio denominator and the
  `k` in `pass@k` all switch to `keep`.
- Without `keep`, `selected` is `None` and `n` stays as given.

## Output

```
{
  "n": 4,
  "num_prompts": 1,
  "prompts": [{"extracted": ["5", "5", "6", null],
               "answers": ["5", "5", "6"],
               "tally": [["5", 2], ["6", 1]],
               "label": "5",
               "ratio": 0.5,
               "rewards": [1.0, 1.0, 0.0, 0.0],
               "truth_rewards": [1.0, 1.0, 0.0, 0.0],
               "metrics": {"label_accuracy": 1.0, "reward_accuracy": 1.0,
                           "majority_voting_reward": 0.5,
                           "ground_truth_reward": 0.5, "pass@4": 1.0}}],
  "metrics": {"label_accuracy": 1.0, "reward_accuracy": 1.0,
              "majority_voting_reward": 0.5, "ground_truth_reward": 0.5,
              "pass@4": 1.0, "majority_ratio": 0.5},
  "selected": null
}
```

- `extracted` lists the raw extraction result per sample (`None` on failure) and `answers`
  holds the votes after dropping `None` and canonicalizing.
- `tally` is a list of `[answer, count]` pairs in first-seen order.
- `selected` is an index list only when `keep` was given.

## Example

```
outputs = ["so \\boxed{5}", "so \\boxed{5}", "so \\boxed{6}", "no box"]
led = solve(outputs, 4, ["5"])
led["prompts"][0]["label"]          # "5"
led["prompts"][0]["ratio"]          # 0.5
led["prompts"][0]["rewards"]        # [1.0, 1.0, 0.0, 0.0]
led["metrics"]["reward_accuracy"]   # 1.0
led["metrics"]["majority_ratio"]    # 0.5
```

## Constraints

- Pure function: no network, no file I/O, no mutation of the inputs; identical inputs must
  produce an identical ledger.
- Returned floats use ordinary division (`sum / count`); comparisons use a `1e-12` tolerance.
- `n < 1`, a length that is not a multiple of `n` (checked before downsampling),
  `keep < 1`, empty `outputs`, or a ground-truth count that differs from the number of
  questions → `ValueError`.
- Overflow caused by `keep > n` → `IndexError`.
