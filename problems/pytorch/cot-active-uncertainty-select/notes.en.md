# Solution Notes

## Approach

The official shizhediao/active-prompt uncertainty_estimation.py (L49-143) plus the output end of answer_extraction (utils.py L248-294):

1. **Upstream out of scope**: GPT k-sampling and answer_extraction (regex for numbers/A-E/yes-no, gsm8k rounding, last_letters tail truncation) — post-extraction answer strings arrive as inputs; the dataloader truncation (dataset_size>1000 takes 1000) and IO stay out;
2. **In scope**: occurrence counting (empty answers to the -10086 sentinel) → numerical variance / global entropy and disagreement → the three sort_by orderings → qes_limit truncation.

Boundary with existing problems:
- **775 cot-auto-cluster-demos**: picks cluster representatives to **assemble demos**, no sampling statistics; this one picks by sampling **disagreement** (the todos ②-vs-① boundary is exactly this);
- **rs-self-consistency / rs-ttrl-pseudo-reward-ledger**: majority voting folds an answer; this one never votes, it only ranks inconsistency;
- **rs-prm-best-of-n**: step-level aggregation producing answers; this one produces an **annotation priority**.

## Key implementation points (official conventions pinned)

- **NO_SOLUTION mixed into the books**: empty answers go to the `-10086` sentinel key, and the numerical variance expansion feeds it **straight into the variance** (float("-10086")) — "a question with no successful samples has a huge variance" is real official behavior, not a bug; generic/strategyqa disagreement also counts it as one distinct key.
- **Two strategyqa special cases**: the occurrence presets the zero-frequency yes/no keys (no entropy contribution but disagreement always ≥2) + disagreement sorting by `abs(yes-no)` **ascending** (the only ascending key; everything else is -x descending). The official try/except fallback `-len(occurrence)` is a dead branch with the preset keys present, kept as-is.
- **Silent skip for variance sorting**: the official `elif sort_by == "variance" and dataset in (...)` has no else — variance + generic keeps the input order rather than raising.
- **scipy entropy convention**: `-Σ(c/n)·ln(c/n)`, zero frequencies contribute 0 (rel_entr(0,·)=0), all-zero nan. No scipy in the judging env; the `math.log` self-implementation is aligned (anchors: entropy([2,2])=ln2, entropy([5])=0).
- **np.var is ddof=0 population variance** (not the sample variance).
- **Stable sorting**: ties keep the input order (dataloader order) in all three keys — python's sort is stable; `reverse=True` also keeps ties stable (it does not reverse-and-concatenate), identical to `-key` (mutation m-sort-tie-rev was removed for this).

## Test design (9 cases)

- `generic_oracle`: 3 questions × k=5 against a Counter-based oracle on three sort_by keys + scattered occurrence/disagreement/entropy assertions;
- `numerical_variance`: NO_SOLUTION entering the variance (-10086 expanded, variance >1.6e7 with the exact formula) + zero variance for unanimous votes;
- `strategyqa_sort`: abs(yes-no) ascending + preset zero frequency excluded from entropy (-(0.8ln0.8+0.2ln0.2)) but included in disagreement (=3);
- `entropy_ledger`: exact descending three-step entropy values (ln5 / -0.6ln0.6-0.4ln0.2 / 0);
- `sort_variants`: variance+generic silently unsorted, entropy descending, numerical+entropy against the oracle;
- `qes_limit`: =2 truncates and sorts within the subset / beyond-count = all / 0 = all / =1 single element;
- `tie_stability`: disagreement ties keep input order + entropy ties likewise;
- `purity`; `args_validation` (types / empty sublists / three enums / limit negative, bool, float + messages).

## Mutation coverage

29 designed → **27 CAUGHT + 2 equivalence removals** (m-limit-cap: `>` to `>=` identical at limit==n; m-sort-tie-rev: python's reverse=True keeps ties in original order, identical to -key).
Coverage: sentinel key dropped / counting not accumulated / first occurrence counted 0 / preset keys dropped; variance ddof=1 / each key expanded once / numerical key widened; entropy base-2 / unnormalized / zero-frequency crash / sign flip; six sorting mutations (dis ascending / sqa difference negated / sqa special-case routing deleted / variance condition widened / entropy ascending); two limit mutations (the 0 semantics dropped / dataset_idx offset); eight validation mutations + two message mutations.

## Engineering remarks

- **Mutation snippet indentation**: 776's validation loop body sits at 12 spaces (inside `for i, samples in enumerate(...)`); writing 16 spaces in the mutation string fails the count=0 assertion — copy replacement strings from the actual file lines.
- **Pin the scipy entropy anchors before self-implementing**: entropy([2,0,3])=0.673, entropy([0,0])=nan (all-zero is not 0) — zero frequency and all-zero are two independent boundaries: the former enters the books (strategyqa preset keys), the latter only appears at num_trails=0 (the nan semantics kept in the implementation while tests avoid it).
- Run the oracle before handwriting expected values (variance >2e7 was a guess, the real value 1.63e7; the entropy_ledger expected order was reversed) — both false failures came from unverified mental arithmetic.
