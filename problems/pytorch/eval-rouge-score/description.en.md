# Evaluation: ROUGE-1 / ROUGE-2 / ROUGE-L

## Problem

ROUGE (Recall-Oriented Understudy for Gisting Evaluation, Lin, ACL 2004) is the workhorse metric for summarization and generation. It does not ask "is what you said correct" but **"how much of the reference content did you cover"** — recall-driven, with an F1 that balances in precision. ROUGE-N counts matching n-grams; ROUGE-L counts the **longest common subsequence** (LCS): the latter is order-sensitive but allows gaps, which is exactly how it divides labour with ROUGE-2 (contiguous vs. ordered). It is the official score of summarization data sets such as CNN/DailyMail and XSum, and rule-based rewards in RLHF frequently plug in ROUGE-L directly.

## Rules

- `hypotheses`: `list[list[int]]` — each candidate summary is a list of token ids (already segmented; no tokenizer here).
- `references`: `list[list[list[int]]]` — `references[i]` is the reference list of hypothesis \(i\) (at least one).
- Aggregation: compute F1 per pair (one hypothesis × one of its references), **take the maximum F1 over that hypothesis's references**, then take the **arithmetic mean over hypotheses**.
- **ROUGE-N** (\(n \in \{1, 2\}\)): with n-gram frequency tables \(\text{count}_h\), \(\text{count}_r\),
  \[ O_n = \sum_{g}\min\big(\text{count}_h(g),\ \text{count}_r(g)\big),\quad
  R = \frac{O_n}{\sum_g \text{count}_r(g)} = \frac{O_n}{\max(0, |r|-n+1)},\quad
  P = \frac{O_n}{\max(0, |h|-n+1)} \]
  \[ F1 = \frac{2PR}{P+R} \]
  Note that \(O_n\) is a **clipped overlap**: n-grams repeated in the hypothesis cannot score beyond their reference frequency.
- **ROUGE-L**: with \(l = \text{LCS}(h, r)\) (you must write the dynamic program yourself),
  \[ R = \frac{l}{|r|},\quad P = \frac{l}{|h|},\quad F1 = \frac{2PR}{P+R} \]
- Boundary conventions (all of them must be handled explicitly):
  - an empty hypothesis or empty reference \(\Rightarrow\) that pair's F1 is \(0\);
  - \(|h| < n\) \(\Rightarrow\) no n-gram of that order exists \(\Rightarrow\) F1 is \(0\) (ROUGE-2 of a one-word summary is necessarily 0);
  - \(P = R = 0\) \(\Rightarrow\) F1 is \(0\); never emit \(0/0\);
  - no hypotheses at all (empty candidate set) \(\Rightarrow\) all three metrics are \(0.0\).
- Return the triple `(rouge1, rouge2, rouge_l)` of Python floats. `rouge_score` / `nltk` are forbidden (and not installed).

## Example

```
hypotheses = [[1,2,4,3], [1,2,5]]
references = [[[1,2,3,4]], [[1,2,3,4,5]]]

hypothesis 0 = [1,2,4,3] vs reference [1,2,3,4]
  ROUGE-1: O=4, R=4/4=1, P=4/4=1              -> F1 = 1
  ROUGE-2: h bigrams {(1,2),(2,4),(4,3)}, r bigrams {(1,2),(2,3),(3,4)} -> O=1
           R=1/3, P=1/3                        -> F1 = 1/3
  ROUGE-L: LCS([1,2,4,3],[1,2,3,4]) = 3, R=3/4, P=3/4 -> F1 = 0.75

hypothesis 1 = [1,2,5] vs reference [1,2,3,4,5]
  ROUGE-1: O=3, R=3/5=0.6, P=3/3=1            -> F1 = 0.75
  ROUGE-2: O=1 (only (1,2) matches), R=1/4, P=1/2 -> F1 = 1/3
  ROUGE-L: LCS = 3 ([1,2,5] is itself a subsequence), R=3/5, P=1 -> F1 = 0.75

averages: rouge1  = (1 + 0.75)/2      = 0.875
          rouge2  = (1/3 + 1/3)/2     = 0.3333333333
          rouge_l = (0.75 + 0.75)/2   = 0.75
```

For hypothesis 0 alone: ROUGE-1 is 1 while ROUGE-2 is only \(1/3\) and ROUGE-L is 0.75 — reordering does not hurt unigrams, it hurts bigrams, and LCS sits in between.

## Constraints

- CPU; token ids are arbitrary non-negative integers, only equality is tested.
- All three returned values lie in \([0, 1]\).
- **Degenerate properties**: hypothesis identical to reference \(\Rightarrow\) all three are 1; disjoint token sets \(\Rightarrow\) all three are 0; duplicating a reference leaves the score unchanged (a maximum is taken, not an average); a one-word summary has ROUGE-2 \(= 0\) while ROUGE-1/ROUGE-L can still be positive.
- Key difference from BLEU: recall and precision appear **together** in the F1, so verbose output hurts precision and terse output hurts recall — no separate brevity penalty is needed.
