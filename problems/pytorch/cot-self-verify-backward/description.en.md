# Self-Verification Backward Checking

## Problem

Self-Consistency voting only counts how often each answer appears; Self-Verification (arXiv:2212.09561) goes further: it **fills each candidate answer back** into the question for backward checking — the numbers in the question are blanked into `'X'` one at a time, a verifier judges whether "the candidate answer still agrees with the context", and agreement with the pre-blank original number earns points; the candidate with the highest total wins.

You are given (GPT sampling, answer_cleansing and the verifier model all live outside the pool): `question` (the raw question), `sampled_answers` (k post-cleansing answer strings), `verifier_outputs` (a flat verdict pool), `ground_truth`, and the optional cached `declarative`.

## The ledger to implement

### 1. Candidate collection

Try `str(float(x))` normalization on each sampled string (`"5"→"5.0"`, non-numeric kept as-is); collect it only if non-empty and **not seen before** (first-seen dedup).

### 2. The declarative rewrite

- Non-empty cache containing `answers[0]`: replace the **last** occurrence of answers[0] with the current candidate (the official reverse-find splicing ≡ `rfind` replacement);
- Empty cache or not containing answers[0]: the fallback template `"{question_tail} The answer is {answer}."` (question_tail is the text after the last period; the GPT regeneration is out of scope).

### 3. Number-blanked subquestions

```
text = question.replace(",", ".")                  # official: commas treated as periods
ts = text[:len(text) - text[::-1].find(".")]       # truncate at the last period (included)
numbers = findall(r"-?\d+\.?\d*", ts)              # negatives and trailing dots ("6." is extracted)
for each number:
    walk the segments of ts.split("."); every segment containing the number
    gets its character range replaced by "'X'" (list slice assignment, first occurrence;
    the official number_find flag is missing, so replacing all matching segments is faithful)
    non-replaced segments get "." appended, then replace("..", ".")
    strip the leading space, pad a trailing space, replace(" .", ".")
    subquestion = '"' + new_ts + declarative + '"'
keep at most the first 3 numbers (official sentences[:3] / ans[:3])
    traps: period-spanning and trailing-dot numbers match no segment → no blanking
```

**Traps**: a number spanning periods (like `-2.5`) matches no single segment → no blanking happens and the sentence stays as-is; a number with a trailing dot (`"6."`) likewise — `"6."` is not inside any dot-less segment.

### 4. Verifier scoring

Convert each verdict string via try-float; convert the expected value via try `float(blanked_number)` (fallback to the raw string, then 0 points); `score_d = sum(v == expected)`; the candidate total is Σ_d score_d. The verdict pool is flat-indexed by `a_idx*3 + d` (at most 3 subquestions per candidate); missing entries score 0.

### 5. Selection and judgement

- Single candidate: `scores = {0: 1}`, no subquestions, no scoring;
- Otherwise argmax(scores), **ties to the earliest** (np.argmax semantics);
- `is_true` = whether each normalized candidate equals the normalized GT; `chosen_correct` = whether the winner hits.

Return `dict(answers, sentences, sub_numbers, scores, chosen_idx, chosen_answer, is_true, chosen_correct)`.

## Input constraints

- `question`/`ground_truth`/`declarative` are strings (question non-empty);
- `sampled_answers` is a non-empty list[str]; `verifier_outputs` is a list[list[str]] (verdicts all strings).

## Hints

- `"-2.5"` is split into `-2` and `5` by the periods — blanking matches **within segments**;
- An empty declarative falls into the "not containing answers[0]" branch and reaches the same fallback (no non-empty string is `in ""`);
- The float normalization in scoring is bidirectional: verdicts and expected values each try-float independently.
