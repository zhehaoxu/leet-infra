# In Depth: Cosine and Hinge Embedding Margin Losses

## Source

- The judging contract is the starter docstring plus the **measured behaviour** of `F.cosine_embedding_loss` / `F.hinge_embedding_loss` (this problem cross-checks against them at atol \(10^{-6}\)).
- Academic origins: the cosine embedding loss is usually traced to Jia et al., *Large-scale Video Classification with Convolutional Neural Networks* (CVPR 2014, which used the angle instead of the dot product to tolerate missing/extra labels); the \(\max(0,m-x)\) shape of hinge embedding is the soft-margin SVM hinge (Cortes & Vapnik, *Support-Vector Networks*, Machine Learning 1995 — a journal article, no arXiv id) in its pairwise Siamese form.
- **Two places where the documentation must not be trusted and the library was measured instead** (all numbers from torch 2.14.1):
  1. `nn.CosineEmbeddingLoss`'s docstring writes the target=−1 branch as \(\max(0,\text{margin}-\cos)\), but the implementation is \(\max(0,\cos-\text{margin})\). Measured with x against itself (\(\cos=1\)), target=−1: margin=0 → **1.0**, margin=0.5 → **0.5**, margin=2 → **0.0**; only the second formula explains that triple. This problem follows the implementation.
  2. `F.cosine_similarity` and `F.cosine_embedding_loss` do **not** use the same eps guard (see below), so the task pins the rule as "clamp the product of the norms as a whole at \(10^{-8}\)", and cross-checks against the library only at normal magnitudes.

## Full derivation and index convention

**Index convention**: \(i\) is pair \(i\); \(\cos_i\) its cosine; \(\text{target}_i\in\{1,-1\}\); both branches are **elementwise**, and reduction happens only at the very end.

**The two cosine branches**: the target=1 branch \(1-\cos\) is 0 at \(\cos=1\) and **2** at \(\cos=-1\) (upper bound 2, not 1); the target=−1 branch \(\max(0,\cos-m)\) is a one-sided hinge that "lets \(\cos\) grow up to \(m\) before paying". The anti-parallel test case lines all of this up at once: three pairs with \(\cos=[-1,1,-1]\), target \(=[1,-1,-1]\), \(m=0\) → \([2,1,0]\). Pair one is "labelled same-class but pointing opposite", so it takes the maximum penalty 2; pair three is "labelled different and already opposite", so it pays exactly 0. That one row verifies both directions (target=1 → \(1-\cos\), target=−1 → \(\cos-m\)) simultaneously.

**Why the angle rather than the dot product**: \(\langle x_1,x_2\rangle=\lVert x_1\rVert\lVert x_2\rVert\cos\theta\), so the raw dot product folds vector length into "similarity"; cosine embedding divides the lengths out and the loss depends only on direction. In Siamese networks this is a real benefit — the same pair is not judged "more similar" merely because the features were globally scaled up. (The price is an extra orthogonal component in the gradient, which is out of scope here.)

**Three ways to write the denominator clamp, and what is observable**: with \(w_i=\lVert x_i\rVert_2\),
 ① clamp the product (this problem): \(\cos=\text{dot}/\max(w_1w_2,10^{-8})\);
 ② clamp each norm: \(\cos=\text{dot}/(\max(w_1,10^{-8})\max(w_2,10^{-8}))\);
 ③ divide only when both norms are strictly positive, else record 0.
Take \(w_1=w_2=10^{-5}\) (parallel vectors, dot \(=10^{-10}\)): in ② neither norm exceeds the floor so nothing triggers → \(\cos=1.0\); in ① the denominator is floored to \(10^{-8}\) → \(\cos=0.01\). Measured: `F.cosine_similarity` returns **1.0** (i.e. form ②), while `F.cosine_embedding_loss` corresponds to an angle of **0.9901** — the two native functions disagree with each other in this region. The judge only exercises \(O(1)\)-magnitude data plus the zero vector, and there ①②③ all reduce to "\(\cos=0\)" (zero numerator, finite denominator), so this problem's guard is **only observable at the zero vector**, where it must give \(\ell(\text{target}=1)=1\) and, for doubly-zero vectors with target=−1, 0 (asserted with `isfinite`).

**Shapes of the two hinge branches**: the target=1 branch is \(x\) itself with **no relu**, so the loss can be negative and is unbounded below as \(x\) grows — the judge pins it with `torch.equal(got, scores)`, and measurement confirms `F.hinge_embedding_loss(x=[-1], target=[1]) = -1.0`. This is not a typo but the definition of `nn.HingeEmbeddingLoss`: for a genuine pair you want the score as large as possible, so the score *is* the loss (minimising it maximises the score). The target=−1 branch \(\max(0,m-x)\) is the textbook hinge: satisfied once the score drops below \(m\), with no further penalty below that.
Do the two branches meet at \(x=m\)? The target=−1 branch is 0 for \(x\ge m\) while the target=1 branch is \(m>0\) there — but they act on **disjoint subsets of samples** (dispatched by target), so no sample crosses over and there is no continuity requirement to satisfy.

**Dispatching the margin default**: `margin=None` resolves to different constants per mode (0.0 / 1.0). This is **not** "one shared default of 0"; each branch must apply its own, matching `cosine_embedding_loss(margin=0)` and `hinge_embedding_loss(margin=1)`.

**The empty set**: the native mean of an empty tensor is NaN (measured: `F.hinge_embedding_loss(zeros(0), zeros(0)) = nan`), while this problem returns scalar 0; sum and none are 0 / (0,) naturally. This is an explicit place where the task overrides the library, recorded here so that a NaN seen while cross-checking is not mistaken for a bug.

## Key points and common mistakes

1. **Direction of the cosine target=−1 branch**: \(\max(0,\cos-m)\), not \(\max(0,m-\cos)\); the margin sequence (0, 0.5, 2) at \(\cos=1\) must give (1, 0.5, 0), whereas the reversed formula is identically 0.
2. **The hinge positive branch is the identity**, negative losses allowed; writing `relu(x)` or `1-x` fails the `torch.equal` probe instantly.
3. **The clamp is on the product** (the only rule here); clamping each norm first differs by two orders of magnitude in extreme regimes.
4. **A zero vector is not an error**: no exception, no NaN — with target=1 the loss is exactly 1.
5. **The two modes have different margin defaults**; using one default for both (e.g. 1.0) breaks the cosine branch.
6. **hinge ignores x2**: whatever is passed must not raise ("ignored ≠ validated"), yet under cosine x2=None must raise ValueError — the same argument is "ignored" in one mode and "required" in the other.
7. **Empty mean returns 0, not NaN**, and must be a 0-dim tensor.
8. **All three reductions**: `"sum"` is legal here (unlike `iou-variant-losses`, where "sum" must raise); don't cross-contaminate the two batches' conventions.
9. **Error surface**: mode ∉ {cosine,hinge}, reduction ∉ {mean,sum,none}, and cosine with x2=None → ValueError.

## Differences from neighboring problems

- **vs `cosine-similarity`**: that problem implements the \(F\).cosine_similarity primitive alone (arbitrary dim, its own eps semantics); here we build the **target-dispatched two-branch loss** on top of it plus reductions, and the denominator rule is stated as its own contract.
- **vs `triplet-margin-mining` (same batch)**: triplets require mining negatives out of the batch, take embeddings+labels as input and must handle empty pools, fallbacks and ties; here the input **is already a given pair**, with no mining, no anchor, and targets restricted to \(\pm1\).
- **vs `contrastive-sampling`**: that problem is about *how to sample* pairs/triplets from labels (the data side); this one is the loss itself (the operator side).
- **vs `mm-clip-infonce`**: InfoNCE normalizes many negatives into a softmax denominator — a "many-to-one" structure; cosine embedding is per-pair independent two-branch hinges with no competing normalization.

## Cross-check convention and complexity

- The primary cross-check is **against the native library**: `F.cosine_embedding_loss` over margin ∈ {0.0,0.2,0.7} × reduction ∈ {none,mean,sum} (16×7 random vectors) and `F.hinge_embedding_loss` over margin ∈ {1.0,0.5} × three reductions (20 random scores). At \(O(1)\) magnitudes these agree elementwise with this problem's contract, so no separate Python oracle is needed — but the **empty mean** and the **tiny denominator** cannot be cross-checked (the native one returns NaN; the native functions disagree with each other), so both are covered by the stated convention plus explicit assertions.
- Discriminating cases: three anti-parallel samples (pinning both branch directions and the target=1 upper bound of 2), target=1 with cos<0 → loss > 1, the margin truncation (cos=1, m=0.5 → 0.5), the all−1 / all+1 hinge branches (the latter asserts the output is **bit-identical** to the input with `torch.equal`), finiteness for zero and doubly-zero vectors, the three N=0 reductions, and four ValueError paths.
- Example constants were backfilled from the reference run and are checkable on paper: both cosine pairs have \(\cos=1/\sqrt{10}\) exactly (dot products both 1, norm products both \(\sqrt{10}\)), so the two branch losses sum to 1.0 — a deliberately constructed self-check.
- Complexity: both branches are \(O(ND)\) (one dot product and two norms) / \(O(N)\) for hinge, plus \(O(N)\) for the reduction; memory is just the intermediates. Nothing here is quadratic in the batch — it is a pure elementwise operator.
