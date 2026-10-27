# In Depth: Structured Dropout: DropPath and Dropout2d/3d

## Source

- Stochastic Depth: Huang et al., *Deep Networks with Stochastic Depth*, ECCV 2016 (**arXiv:1603.09382**) — during training an entire residual branch is set to identity with some probability (a per-sample Bernoulli coin), with the survival probability linearly decaying with depth; DropPath is the minimal implementation unit of that mechanism: a single layer of "whole-sample dropping + rescaling".
- timm's DropPath implementation convention (huggingface/pytorch-image-models, drop_path in `timm.layers`): keep_prob = 1−p, mask = floor(keep_prob + rand\((B,1,\dots,1)\)), y = x.div(keep_prob) · mask — the same formula as this problem's \(y=x\cdot mask/(1-p)\); the only difference is whether the mask is RNG-sampled (timm) or passed explicitly (this problem).
- Dropout2d / Dropout3d: the official torch documentation — "Randomly zero out entire channels" (each feature map \(\text{input}[i,j]\) of each sample zeroed independently); the torch docs also cite Tompson et al., *Efficient Object Localization Using Convolutional Networks* (2015, **arXiv:1502.02604**) as the origin of spatial dropout: when adjacent pixels within feature maps are strongly correlated, i.i.d. dropout does not regularize and merely acts as an effective learning-rate decrease, so whole maps must be dropped.
- Inverted scaling is the standard training-phase bookkeeping of the dropout family (consistent with `F.dropout` training=True): with mask ~ Bernoulli(\(1-p\)), \(E[y]=x\cdot(1-p)/(1-p)=x\), and no rescaling is needed at inference.

## Full derivation and index convention

**One formula, three granularities**. Write \(x\ (N,C,S_1,\dots,S_k)\) and the unified form \(y[b,c,\dots]=x[b,c,\dots]\cdot m[b,c]/(1-p)\); the three modes differ only in m's broadcast shape:

- droppath: \(m\) of shape \((B,1,\dots,1)\) — one coin per sample, all elements of a sample live or die together (timm's per-sample stochastic depth);
- dropout2d: \(m\) of shape \((N,C,1,1)\) — one coin per sample per channel, the whole \(H\times W\) feature map zeroed;
- dropout3d: \(m\) of shape \((N,C,1,1,1)\) — the channel block includes the depth dim, the whole \((D,H,W)\) block zeroed.

Dropped blocks are strictly zero (\(0\cdot x=0\), no approximation) and surviving blocks are exactly \(x/(1-p)\). The mask allows two equivalent forms — compact \((B,)\)/\((N,C)\) and all-ones \((B,1,\dots,1)\)/\((N,C,1,\dots,1)\); reshape to the all-ones form and multiply, and the two forms are `torch.equal`-equivalent (asserted by the judge).

**Why the whitelist is "broadcast must equal x's shape exactly"**: broadcasting would also accept \((1,C)\), \((B,1)\), etc., but those mean different granularities ("channels sharing a coin across samples", "channels split further") — this problem pins the granularity and rejects any extra/missing dim with `ValueError`. droppath is generic over arbitrary tail shapes of rank \(\ge1\) (\((B,)\) always broadcasts a tail of 1s); 2d/3d require exactly 4/5 dims and reject a wrong rank first.

**The boundary account for p**: p=0 → scale=1, \(y=x\cdot mask\); with an all-ones mask this is a **strict identity** (`torch.equal` requires bit-level agreement — the implementation must not take any path that changes dtype or introduces numeric noise). \(p\ge1\) → \(1/(1-p)\) divides by zero or goes negative, which has no semantics; \(p<0\) and non-finite p (nan/inf) likewise — nan's comparisons are always False, so an explicit isfinite check is required. `F.dropout`'s valid domain is also \([0,1)\).

**Expectation bookkeeping and "who scales"**: treating the mask as a realization of Bernoulli(\(1-p\)) gives \(E[y]=x\) — that is the inverted convention (divide by the survival rate during training) versus the old "multiply by \((1-p)\) at inference". With the mask given explicitly the formula degenerates to deterministic elementwise scaling, but the **scaling convention still follows the training account**: the scale depends only on p, never on the mask contents (the judge asserts surviving blocks equal \(x/(1-p)\)).

**bool mask semantics**: True=1=keep, False=0=drop. Per the starter's convention, cast to x's dtype before multiplying — the exact same path as a 0/1 float mask (the judge asserts the two types are `torch.equal`), which also guarantees the output dtype matches x.

## Key points and common mistakes

1. **Elementwise instead of block granularity**: expanding the mask to x's full shape and multiplying elementwise is the most common mistake — the judge checks per block ("dropped block entirely 0 / surviving block entirely \(x/(1-p)\)").
2. **Missing or misplaced scaling**: \(y=x\cdot mask\) (missing scale), \(y=x\cdot mask\cdot(1-p)\), \(y=x\cdot mask/p\) are all wrong; scale \(=1/(1-p)\) depends only on p.
3. **The mask shape whitelist**: a 3-D x under droppath accepts only \((B,)\)/\((B,1,1)\); \((B,1,1,1)\), \((B,C)\), \((B,1)\) all raise `ValueError`. The intermediate form \((N,C,1)\) is illegal for 2d/3d too.
4. **p validation includes non-finite values**: `float("nan")`/`float("inf")` do not satisfy \(0\le p<1\) (nan comparisons are always False; isfinite is mandatory); at p≥1 the \(1/(1-p)\) divides by zero.
5. **"Strict" in the p=0 identity**: `torch.equal`, not allclose; no extra dtype casts or clamps.
6. **Return a new tensor**: do not multiply x in-place.
7. **No RNG inside the function**: the mask is given; calling `torch.rand` internally breaks both the determinism contract and the division of labor with the `dropout` problem (seed-built mask).
8. **Rank checks first**: dropout2d receiving a 3-D/5-D x raises `ValueError` — never silently treat it as droppath; modes outside the whitelist (e.g. `"DropPath"`, `"dropout1d"`) raise `ValueError` too.

## Differences from neighboring problems

- **`dropout`**: elementwise independent Bernoulli, mask built internally from a seed via `torch.Generator`, judged by full-tensor comparison; here blocks are dropped (whole samples/channels), the mask is an explicit argument, and there is no RNG at all. Both share the inverted-scaling bookkeeping, but mask granularity and randomness source differ — "build the mask from a seed" (`dropout`) vs "the mask is the input" (this problem) is a deliberate division of labor.
- **`batchnorm-forward`**: another dimension of the train/eval state — it switches batch statistics vs running statistics with a training flag; this problem is the forward of the training-phase regularizer itself, with the inference shape (p=0, all-ones mask) as a mere degenerate point and no branching logic.
- **`F.dropout2d` training=False**: returns identity without scaling; this problem always scales by the given p — "whether anything is dropped" is decided by the mask and "whether to scale" by p, decoupled.
- **timm DropPath**: the training-mode implementation with an RNG-sampled mask; this problem turns it into a deterministic operator (mask as input), with the formula identical to timm's `x.div(keep_prob)·mask`.

## Cross-check convention and complexity

- **Independent oracle**: pure-Python recursion/loops performing manual broadcasting plus elementwise \(\cdot m\cdot\) scale, on a separate path from the tensorized solution — droppath recurses over arbitrary ranks, 2d/3d iterate per \((n,c)\) block; `allclose` 1e-6 (1e-12 for float64), with shape and dtype asserted one by one.
- **Discriminating probes**: dropped blocks `torch.all(==0)`, surviving blocks `==x/(1-p)`; \((B,)\equiv(B,1,\dots,1)\); bool ≡ 0/1 float; all-ones mask → \(y=x/(1-p)\); the strict p=0 identity (`torch.equal`); the full `ValueError` taxonomy over p / mask shape / mode.
- **Deterministic data**: golden-ratio-hash pseudo-random tensors and thresholded bool masks (no RNG), covering p ∈ {0, 0.25, 0.5, 0.875}, ranks 2/3/4/5, float32/float64.
- **Complexity**: time \(O(\text{numel})\) — one broadcast multiply (the mask has far fewer elements than numel) plus one scalar scaling; whitelist validation \(O(1)\). Space \(O(\text{numel})\) (a new tensor).
