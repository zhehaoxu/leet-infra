# In Depth: Soft-NMS with Continuous Confidence Decay

## Source

- Bodla et al., *SoftNMS: Improved Non-Maximum Suppression Algorithm* (ICCV 2017, arXiv:1704.04503). The paper's motivation is that hard NMS eats the second ground-truth box whenever two real objects sit **close together** (mutually occluded pedestrians); the soft version replaces "delete the box" with "continuously suppress the confidence by overlap" and offers two decay kernels, linear (piecewise) and gaussian.
- The exact constants and formula shapes are defined solely by the starter docstring: the linear kernel is gated by a **strictly greater** hard threshold and multiplies by \((1-\text{IoU})\); the gaussian kernel multiplies by \(\exp(-\text{IoU}^2/\sigma)\) unconditionally and never consults the threshold. This two-kernel shape matches the `SOFT_NMS` post-processing found in detection frameworks such as detectron2; the paper writes the gaussian temperature as \(t\), while this repo keeps the more common \(\sigma\).
- The IoU kernel reuses this repo's `iou-compute` convention; no coordinate-format conversion (cxcywh is out of scope).

## Full derivation and index convention

**Index convention**: original indices \(i\in\{0,\dots,N-1\}\), current score vector \(s\in\mathbb R^N\) (initialised to the input scores), active set \(A\) (initially everything). Round \(k\) selects \(\arg\max_{i\in A}s_i\) (ties → smallest \(i\)); the kept sequence is \(M\).

**Exact single-round update**: with \(b\) selected this round, for every \(j\in A\setminus\{b\}\):
\[\text{linear: } s_j\leftarrow s_j\cdot\begin{cases}1-\mathrm{IoU}(b,j),&\mathrm{IoU}(b,j)>\text{iou\_thr}\\[2pt]1,&\text{otherwise}\end{cases}
\qquad
\text{gaussian: } s_j\leftarrow s_j\cdot e^{-\mathrm{IoU}(b,j)^2/\sigma}\]
then \(A\leftarrow A\cap\{j:s_j\ge \text{score\_thr}\}\), and \(b\) is removed from \(A\) (**entering the output also means leaving the pool — it never reappears**).

**Why it always terminates**: each round shrinks \(A\) by at least one element (the selected \(b\)), and culling only shrinks it further, so there are \(\le N\) rounds. The decay factor always lies in \([0,1]\) (linear's \(1-\text{IoU}\) is exactly 0 when IoU=1, and a zeroed score is then culled by `score_thr`; gaussian's exponential is always positive), hence every \(s_j\) is **monotonically non-increasing** — which proves "score at selection ≤ original score" and is why `score_thr` needs checking only once, after decay.

**Why the output order is not the input score order**: the example returns `keep_indices=[0,2,1]`, not `[0,1,2]`. Box 1 is suppressed to 0.2555 in round 1, dropping below the originally lower box 2 (0.7), so round 2's winner changes. **This is precisely soft-NMS's signature behaviour**: a low-scoring but isolated box can be emitted before a high-scoring box that overlaps something already chosen. Hard NMS never does this — it rewrites no values, so its output order is always the survivors' original score order.

**How the two kernels differ in shape**: linear is **piecewise** — below (or at) the threshold the factor is exactly 1 (nothing changes), and it jumps to \(1-\text{IoU}\) only past it; this keeps hard NMS's "low overlap is not punished" intuition at the price of discontinuity at the threshold (the two test cases at 0.5/0.5 and 0.499999 sit exactly on that discontinuity). gaussian decays everywhere with no threshold: IoU=0 gives exactly \(e^0=1\), IoU=1 gives \(e^{-1/\sigma}\) (0.1353 at \(\sigma=0.5\)) — **never 0** — so under gaussian even a perfectly coincident box can survive on a tiny score unless `score_thr` removes it.

**Ties and \(-\infty\) masking**: filling inactive slots with \(-\infty\) makes `argmax` return the **first** maximal index, which is exactly "ties take the smallest original index". Equivalently, the oracle scans left to right and updates `best` only on a strict `>`; two implementations of one semantics. Note that \(-\infty\) must appear only in masked slots, and the loop guard must stop calling argmax once \(A=\varnothing\) (with everything \(-\infty\), argmax still returns 0 and would re-select a deleted box).

**`score_thr` is not a pre-filter**: the test suite contains \(N=1\) with scores \([0.0]\), which **must** return `[0]`. So `score_thr` only acts "after some round's decay"; the first-round winner is always emitted — matching the docstring's "with N≥1 the first top box is always kept". Implementing it as "delete everything below the threshold first, then iterate" fails outright.

**Culling must be `active &=`**: writing the threshold test as `active = cur >= score_thr` (assignment instead of AND) makes an already-selected box **resurrect**, because its score is still high, and it gets emitted twice — the most common out-of-bounds bug in this family of implementations. Likewise decay should apply only to active slots (`torch.where(active, ...)`); it does not change the output either way, but it silently rewrites culled boxes' scores and hides the bug above.

## Key points and common mistakes

1. **Record first, then decay**: the emitted score is the value "at the moment of selection"; reversing the order multiplies the selected box by its own factor too.
2. **Linear is strictly greater**: \(\text{IoU}=\text{iou\_thr}\) does not decay (pinned by a box pair whose IoU is exactly 0.5).
3. **Gaussian ignores iou_thr**: IoU≈0.005 must still decay; conversely IoU=0 gives a factor of exactly 1, so disjoint scores are bit-for-bit unchanged (`torch.equal` level).
4. **"Ignored" is not "validated"**: under linear, sigma=-5 must still return normally.
5. **`score_thr` deletes only on strictly less**: equality is kept.
6. **Selected means out of the pool**, forever; culling can only shrink the active set.
7. **N=0** returns two (0,) empty tensors of long and float — dtype is checked too.
8. **Error surface**: illegal method ValueError; gaussian with sigma≤0 ValueError; other parameters being nonsense (e.g. passing iou_thr under gaussian) must not raise.

## Differences from neighboring problems

- **vs `nms` (the hard version in this repo)**: hard NMS greedily walks the score-descending list and **deletes** boxes once `IoU > iou_thresh`, returning indices only with untouched scores; soft-NMS keeps the same "strictly greater" suppression condition but replaces deletion with multiplying by \((1-\text{IoU})\), therefore also returning the score at selection time, and its output order may differ from the original score order. The example contrast in this batch (hard `[0,2]` vs soft `[0,2,1]`) is the discriminating anchor.
- **vs `iou-compute` / `iou-variant-losses`**: those compute IoU itself (or its regression loss); here IoU is a **scoring function** used to modulate confidence — the score side, not the coordinate side. The IoU kernel matches `nms` and `iou-variant-losses`, so they can be read side by side.
- **vs `topk-select` / `sort-argsort`**: the output order here is **not** a sorting of the scores (see "why the order changes"), so it cannot be approximated by "sort + truncate" — the judged `[0,2,1]` lies outside the original score order `[0,1,2]`.

## Cross-check convention and complexity

- The oracle `_oracle` is a **structurally different implementation**: Python lists plus an explicit while loop, element-by-element max scan, per-box multiplication, per-box threshold test; IoU comes from `_py_iou`, a scalar version built on builtin `max/min`. Index sequences are compared exactly via `tolist()` (pinning both order and tie semantics); scores with \(10^{-5}\) allclose.
- 12 fixed-seed random box sets × {linear, gaussian} × {defaults, custom iou_thr=0.55 / sigma=0.3 / score_thr=0.05}, covering both extremes of "low threshold, lots of decay" and "high threshold, almost no decay".
- Discriminating cases test semantics rather than numbers: mutually disjoint boxes → output is exactly the original score order with bit-identical scores (`torch.equal`); a pair with IoU exactly 0.5 → the decay/no-decay boundary for thresholds 0.5, 0.3 and 0.499999; tiny overlap (IoU≈0.005) → a **paired assertion** that gaussian decays while linear does not; extreme `score_thr` → only the first-round winner survives; plus \(N\in\{0,1\}\) and the illegal method/sigma error surface.
- Complexity: the precomputed IoU matrix costs \(O(N^2)\) time and memory; each round is \(O(N)\) to select plus \(O(N)\) to decay, over \(\le N\) rounds → \(O(N^2)\) total. Recomputing IoU every round inflates the constant by a factor of \(N\) but stays the same order; the pure-Python oracle is around \(O(N^3)\) (its scalar IoU function is re-invoked for every box each round), which is why it only runs on small cases.
