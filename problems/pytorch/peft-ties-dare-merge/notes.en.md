# PEFT: DARE Drop-and-Rescale with TIES Merging — In Depth

## 1. Task vectors and the two papers

The common language of model merging is the **task vector** \(\delta=\theta_\text{ft}-\theta_0\) (the TIES paper, Yadav et al., arXiv:2306.01708). This problem's inputs are directly \(\delta_1,\delta_2\), testing two operator families on top of them:

- **DARE** (Yu et al., arXiv:2311.03099): randomly drop a \(p\) fraction of parameters and rescale the survivors by \(1/(1-p)\) to preserve the expectation — in theory, "homologous models can drop the vast majority of their redundant increments and still merge well";
- **TIES**: TRIM (keep top-k by magnitude) → ELECT SIGN (majority vote on the sign) → DISJOINT MERGE (average only sign-consistent members).

The DARE-TIES pipeline = DARE first, then feed the residuals to TIES — a common combination in toolchains like mergekit. The judging environment has no third-party merging library; everything is implemented with pure tensor ops per the statement's conventions.

## 2. Three convention pitfalls (judging focus)

1. **DARE's rescaling applies only to survivors**: \(\delta'=m\odot\delta/(1-p)\). Rescaling "at the mean level" or dividing the whole vector by \((1-p)\) still gives 0 at dropped positions (0 times anything is 0) — what truly breaks is writing \(1/(1-p)\) as \(1-p\) or dropping the rescale entirely.
2. **TIES' TRIM is per-vector**: \(\delta_1'\) keeps its own top-`keep_n`, \(\delta_2'\) keeps its own, each flattened and taken by \((|\cdot|\) descending, index ascending\()\). Non-deterministic tie-breaking would make two legal implementations disagree.
3. **The denominator of DISJOINT MERGE**: the paper's Algorithm 1 uses "the number of tasks agreeing with the elected sign" \(|T_j|\), **not** the total task count. With two tasks, a single survivor takes its original value (denominator 1), double agreement takes the average (denominator 2), and a conflict \(\gamma=0\) is zeroed. `check_single_survivor_disjoint_mean` tests exactly this.

## 3. Numerics and determinism design

- Real DARE has \(m\sim\mathrm{Bernoulli}(1-p)\) random; the judge makes the mask an **explicit input** (the collection's determinism convention), so `p` acts only through the rescale.
- TRIM uses `sorted(key=(-abs, index))` (Python's stable sort) for smaller-index tie-breaking; the oracle picks the maximum one at a time with a **selection sort**, structurally independent.
- The degenerate setting \(p=0\), all-ones masks, `keep_n=N` shrinks TIES to "average where signs agree / zero on conflict / original value for a single survivor", all hand-checkable.
- A pair of opposite signs with equal magnitude (\(\delta_1=+5,\delta_2=-5\)) has cancelling votes \(\gamma=0\) → TIES zeroes it, while the DARE mean is also 0 — the two operators converge at "complete interference", the minimal counterexample behind the TIES paper's motivation (sign interference is the main cause of plain averaging's degradation).

## 4. Division of labor within the collection

- `peft-lora-merge`: folding a single adapter's \(W_0+sBA\), no multi-task vectors involved.
- The multi-adapter batch hot-swap entry (item #83 of the backlog): the **forward** synthesis of multiple adapters within a batch, a runtime behavior.
- This problem: **offline weight merging** (after training, before deployment), outputting the merged matrix itself.
