# In Depth: ROUGE-1 / ROUGE-2 / ROUGE-L

## Origin
- Chin-Yew Lin, *ROUGE: A Package for Automatic Evaluation of Summaries* (ACL 2004 Text Summarization Branches Workshop) and Lin & Hovy, *Automatic Evaluation of Summaries Using N-gram Co-Occurrence Statistics* (NAACL 2003); the full definitions are in Lin's 2004 CMU PhD thesis.
- The ROUGE family: \(R_n/P_n/F_n\) (ROUGE-N), ROUGE-L (longest common subsequence), ROUGE-W (weighted LCS), ROUGE-S (skip bigrams). This problem implements the two most used members: ROUGE-1/2 and ROUGE-L.
- Lin combines precision and recall through the weighted harmonic mean \(F_{meas} = \frac{(1+\alpha)RP}{R+\alpha P}\), where \(\alpha\) emphasises recall; \(\alpha = 1\) is the plain F1 used here, which is also the default output of the open-source packages.
- Common implementations: the official Perl package `ROUGE-1.5.5.pl`, `pltrdy/rouge-score`, `torchmetrics.text.ROUGEScore`, `nltk.metrics` (a simplified version only). None of them exists in the judge environment, so everything has to be written from the definition.

## Per-reference maximum vs. pooled references: where implementations diverge
With multiple references the packages disagree. The official Perl package, for ROUGE-N, pools **all references of one hypothesis into a single "model"** frequency table, so the recall denominator is the total n-gram count of every reference; `rouge-score` and `torchmetrics` score each reference separately and **take the maximum**. The two give visibly different numbers when the references differ a lot. To keep the worked example hand-checkable and the result deterministic, this problem uses **per-reference maximum F1**, then averages over hypotheses — stated in the rules, so do not quietly swap in pooled or averaged references.

## Definitions and recurrences
Let \(\text{count}_x(g)\) be the number of occurrences of n-gram \(g\) in sequence \(x\), and \(|\cdot|\) the length.

**ROUGE-N (one pair)**
\[ O_n=\sum_{g\in h}\min(\text{count}_h(g),\text{count}_r(g)),\quad
R_n=\frac{O_n}{\sum_g \text{count}_r(g)},\quad
P_n=\frac{O_n}{\sum_g \text{count}_h(g)},\quad
F_n=\frac{2P_nR_n}{P_n+R_n} \]
The denominator is \(L-n+1\) (zero terms if the sequence is shorter than \(n\), in which case F1 is defined as 0). Clipping caps how often a repeated n-gram can be credited: this is the same idea as in `eval-bleu-score` but used differently — BLEU takes the maximum reference frequency across that sentence's references, clips against it and never looks at recall, whereas ROUGE clips inside a single reference and reports recall as well.

**ROUGE-L**: let \(l(i,j)\) be the LCS length of \(h_{1..i}\) and \(r_{1..j}\), with the standard recurrence
\[ l(i,j)=\begin{cases} l(i-1,j-1)+1, & h_i = r_j\\ \max\big(l(i-1,j),\ l(i,j-1)\big), & h_i \ne r_j \end{cases},\qquad l(0,\cdot)=l(\cdot,0)=0 \]
then \(R = l(|h|,|r|)/|r|\), \(P = l(|h|,|r|)/|h|\) and F1 as above. LCS may match with gaps, so it measures "how long a correctly-ordered backbone is", which is not the longest common **substring** — replacing the \(\max(l(i-1,j), l(i,j-1))\) branch with 0 silently turns the metric into the substring version and makes scores mysteriously low.

## Core points and pitfalls
1. **Clipped overlap is not "number of matched distinct grams"**: each distinct gram contributes \(\min(\text{count}_h, \text{count}_r)\), so a word appearing once in the reference can be credited at most once even if the hypothesis repeats it five times.
2. **The denominators count n-gram occurrences**: \(|r|-n+1\), not the number of distinct n-grams; when \(|h| < n\) the hypothesis side has none, so F1 must be explicitly 0 rather than \(0/0\).
3. **Division when \(P+R=0\)**: disjoint sequences make both zero; F1 is defined as 0 there (not NaN, and it must not raise).
4. **LCS indexing**: the DP table needs \((|h|+1)\times(|r|+1)\) with row/column 0 zeroed; writing it 0-based without the padding row drops the boundary case. Space can be squeezed into two rows, but at this problem's size the full table is clearer.
5. **The average runs over hypotheses**: an empty hypothesis is still a hypothesis and contributes 0 — do not remove it from the denominator, otherwise the catastrophic "model emitted nothing" case gets averaged away. This is the opposite of `eval-token-accuracy`, where samples without valid positions are excluded: there the unit being averaged is a *valid token*, here it is a *sample*.
6. **ROUGE-L does not substitute for ROUGE-2**: with a reordering that keeps every word, ROUGE-1 stays 1, ROUGE-2 collapses and ROUGE-L sits in between — the worked example's hypothesis 0 is exactly 1 / 0.3333 / 0.75. Reporting all three is the summarization convention.
7. **No stemming or stop-word lists here**: the official package supports `--stem` and stop-word files, but they need external resources that the judge environment does not have; inputs are id lists and comparison is plain equality.

## Neighbouring metrics
- **vs `eval-bleu-score`**: BLEU is precision plus a brevity penalty with no recall term at all; ROUGE blends precision and recall into F1. A too-short translation is penalised in BLEU by the BP and in ROUGE-1 by the recall term — different mechanisms, and both metrics aggregate at corpus/sample level.
- **vs `eval-token-accuracy`**: token accuracy compares position by position and requires an alignment of equal length; ROUGE-L matches with gaps, so it can compare sequences of different lengths.
- **vs `eval-shingle-contamination`**: there, Jaccard and containment measure set overlap symmetrically or with a fixed audit direction for contamination screening; here \(R\) and \(P\) are directional with respect to what counts as ground truth — swapping the denominators changes the meaning.
- **vs `retrieval-rrf-fusion`**: RRF fuses rank lists and never inspects content overlap; ROUGE is entirely content overlap.

## Verification approach
The reference solution builds n-gram frequency tables with `Counter`, computes LCS with a full \((|h|+1)\times(|r|+1)\) DP matrix, and handles the zero denominators with explicit branches. The judge's `_oracle` takes a completely different path: n-grams assembled by manual slicing into lists, frequencies counted with `list.count` (no `Counter`, no hash tables), LCS computed by **memoized top-down recursion** instead of bottom-up table filling, and F1 from a separate helper. Besides the worked example the suite asserts: identical strings → 1, reordering keeps unigrams but breaks bigram/LCS, empty hypothesis, disjoint sets → 0, one-word hypothesis → ROUGE-2 is 0, multi-reference picks the maximum, duplicated references are inert, empty candidate set → all zeros, plus randomised corpora cross-checked end to end, all with `allclose(atol=1e-5, rtol=1e-5)`.

## Complexity
With total hypothesis length \(T\) and reference length \(S\): ROUGE-N costs \(O(T + S)\) n-gram construction and lookup (tuple construction is \(O(n)\)); ROUGE-L costs \(O(|h||r|)\) time per pair and the same order of space (reducible to \(O(\min(|h|,|r|))\)).

## Practical notes
ROUGE is lenient towards long summaries (writing more means hitting more), which is exactly why models optimise it as a rule-based reward by padding output. Report \(P, R, F\) together, and fix the number of references and the aggregation convention — otherwise a 0.02 difference across papers means nothing.
