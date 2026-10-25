# Sampling: Contrastive Search with Degeneration Penalty (CNTRM)

## Problem Statement

Greedy and beam search drag autoregressive models into **degeneration**: the sentences stay grammatical yet
repeat the same handful of words, and the information content decays with length. Pure random sampling does
break the loop, but at the cost of semantics drifting off topic. **Contrastive Search** (CNTRM in the paper)
rewrites "which token comes next" as a **two-objective** decision: the chosen token must both be probable
according to the model *and* be far enough from what has already been written **in representation space**. A
decoding step therefore no longer compares \(\log p\) alone but "confidence minus a degeneration penalty",
taking an **argmax** over the \(\top_k\) candidate set.

The **source of the penalty** is where this problem parts ways with every repetition-suppression sampler.
`penalty-sampling` only counts how many times a token appeared inside the history window — string-level
bookkeeping. Here the penalty comes from the **cosine similarity** between the hidden state obtained by
appending the candidate to the prefix and the hidden states of the history, so a paraphrased echo is
suppressed just as hard. The similarity is not computed by a model — there is no language model in the
grading environment; it is handed in explicitly as the argument \(sim\), which makes the whole task a pure
tensor computation.

Note also that this problem **takes no random numbers**. Because the selection is an argmax, the same input
always yields the same output for free; that is its most direct interface difference from the "lottery"
samplers of this collection (top-k / top-p / XTC all require uniform random numbers passed in as parameters).
One call handles **one** decoding step: in real decoding the hidden state of the chosen token joins the
history at the next step, so the degeneration penalty rises as repetition accumulates. That cross-step effect
is not simulated here — it is simply reflected in the values of the given \(sim\).

## Rules

Let \(V\) be the vocabulary size, \(z\in\mathbb{R}^{V}\) the logits, \(T\) the number of history hidden states,
and \(\lambda\in[0,1]\) the mixing weight.

- **① Candidate set.** With \(p=\operatorname{softmax}(z)\), take the \(K=\min(k,V)\) tokens with the **largest**
  probability, ties broken by **ascending** token index (stable) → \(\text{cand}_0,\dots,\text{cand}_{K-1}\).
  \(k>V\) is clamped to \(V\), it is not an error; \(k\ge 1\).
- **② Degeneration penalty.** Row **\(j\)** of \(sim\in\mathbb{R}^{K\times T}\) must correspond to candidate
  **\(j\)** (the descending-projection order), not to the first \(K\) rows of the vocabulary:
  \[
  \deg_j=
  \begin{cases}
  \operatorname{clamp}_{[0,1]}\!\Big(\max_{1\le i\le T}\frac{sim_{j,i}+1}{2}\Big), & T>0,\\[6pt]
  0, & T=0\ \text{(no history: }sim\text{ is a }K\times 0\text{ empty matrix).}
  \end{cases}
  \]
  \(\deg_j\in[0,1]\): perfectly aligned with some history state (\(\cos=1\)) scores \(1\), exactly opposite
  scores \(0\). The age order inside the history is irrelevant because only the \(\max\) is taken — the penalty
  is set by the **single most similar** state, not by an average.
- **③ Selection score.**
  \[
  \text{score}_j=(1-\lambda)\cdot\big[\log\operatorname{softmax}(z)\big]_{\text{cand}_j}\;-\;\lambda\cdot\deg_j .
  \]
  The log probability uses the **full-vocabulary** \(\log\operatorname{softmax}\); there is **no renormalization
  inside the candidate nucleus** (unlike top-k sampling, which divides the nucleus down to sum 1).
- **④ Selection.** \(j^{*}=\arg\max_j \text{score}_j\); on an exact tie take the candidate that comes **earlier**
  in the candidate sequence (i.e. the one with the larger logprob), so \(\text{chosen}=\text{cand}_{j^{*}}\) is a
  **vocabulary index**. There is **no lottery anywhere**; a random-number argument would be both superfluous
  and wrong.
- **Return the 4-tuple** \((\text{cand\_idx},\ \deg,\ \text{scores},\ \text{chosen})\): `cand_idx` is `int64`
  of shape \((K,)\); `deg` and `scores` are `float32` of shape \((K,)\) in exactly candidate order; `chosen` is a
  scalar tensor.

## Example

```
Input: z = [1.5, 1.2, 0.5, -1.0] (V=4), k = 3, λ = 0.6, T = 2
  softmax(z)      = [0.456458, 0.338152, 0.167921, 0.037468]
  log_softmax(z)  = [-0.784259, -1.084259, -1.784259, -3.284259]
  candidates (descending top-3) = [0, 1, 2], sim rows aligned with them:
      sim = [[ 0.95,  0.90],   → deg_0 = max(0.975000, 0.950000) = 0.975000
             [-0.80, -0.60],   → deg_1 = max(0.100000, 0.200000) = 0.200000
             [ 0.00,  0.20]]   → deg_2 = max(0.500000, 0.600000) = 0.600000
      (sim lies inside [-1,1], so clamping to [0,1] changes nothing)
  score_0 = 0.4·(-0.784259) − 0.6·0.975000 = -0.898704
  score_1 = 0.4·(-1.084259) − 0.6·0.200000 = -0.553703
  score_2 = 0.4·(-1.784259) − 0.6·0.600000 = -1.073704
  argmax → rank j* = 1 inside the candidate set → chosen = cand_1 = 1   ← the most probable token 0 is pushed out by its penalty
Output: cand_idx = [0, 1, 2]
        deg      = [0.975000, 0.200000, 0.600000]
        scores   = [-0.898704, -0.553703, -1.073704]
        chosen   = 1
```

Keeping the data and only moving \(\lambda\) shows that this is a **continuous trade-off axis**: at
\(\lambda=0\), `scores` are exactly \([\log p_0,\log p_1,\log p_2]=[-0.784259,-1.084259,-1.784259]\) and token
\(0\) wins (the method degenerates into greedy); at \(\lambda=0.2\), scores \(=[-0.822407,-0.907407,-1.547407]\)
and \(0\) still wins; at \(\lambda=0.4\), scores \(=[-0.860555,-0.730555,-1.310555]\) and token \(1\) takes over.
The exact crossover is \(\lambda^{*}=0.3/(0.3+0.775)=0.279070\) (the numerator is the logprob gap, the
denominator adds the degeneration gap on top of it).

```
Extreme λ = 1: only "who is least similar to the history" matters, the candidate set itself is unchanged
  z = [2.0, 1.0, 0.0, -1.0], k = 4, λ = 1.0, T = 2
  sim = [[ 1.0,  0.8],   → deg = [1.000000, 0.250000, 0.700000, 0.400000]
         [-1.0, -0.5],      scores = −deg = [−1.000000, −0.250000, −0.700000, −0.400000]
         [ 0.0,  0.4],   chosen = cand[argmin deg] = 1
         [-0.2, -0.9]]
```

```
Three independent boundary inputs: no history / clamping / ties
  T=0: z = [0.2, 1.9, -0.4, 0.7], sim has shape (4, 0), k = 4, λ = 0.5
       cand = [1, 3, 0, 2], deg = [0, 0, 0, 0],
       scores = [-0.230020, -0.830020, -1.080020, -1.380020], chosen = 1 (identical for λ = 0/0.5/1)
  k>V: z = [1.0, 1.0, 0.0], k = 99 → K = 3, cand = [0, 1, 2] (the 1.0 tie is broken by ascending index)
       sim = all-zero (3, 2) → deg = [0.5, 0.5, 0.5]; λ = 0.5 →
       scores = [-0.680997, -0.680997, -1.180997], ranks 0 and 1 tie exactly → chosen = 0 (earlier candidate)
```

## Constraints

- **CPU only, everything hand-written.** The grading environment `.venv-eval` contains only `torch` / `numpy` /
  `transformers` and is **offline**: no llama.cpp, no HF `LogitsProcessor` / `GenerationMixin` sampler classes to
  borrow, and no `sklearn` or `nltk` either.
- **All randomness must come from explicit arguments** — and this problem simply **has no random-number
  argument**: a single CNTRM step is an argmax over the candidate set, deterministic by construction. Do not call
  `torch.rand` / `manual_seed`, and do not depend on anything beyond the documented tie-breaking of the sort.
- Shapes and dtypes: \(z\) is a 1-D `float32` \((V,)\); \(sim\) is `float32` \((K,T)\) with \(K=\min(k,V)\), whose
  entries are cosine similarities in \([-1,1]\); \(T\) may be \(0\). `cand_idx` / `chosen` are `int64`,
  `deg` / `scores` are `float32`.
- Ranges: \(k\ge 1\) (\(k>V\) allowed, clamped to \(V\)); \(\lambda\in[0,1]\); \(\deg\in[0,1]\).
- Grading tolerances: `deg` with `atol=1e-6`, `scores` with `atol=1e-5, rtol=1e-5`, while `cand_idx` / `chosen`
  must compare **exactly equal**.
- **Identities that always hold**: \(\lambda=0\) ⇒ exactly the most probable token inside the \(\top_k\) = the global
  argmax, and `scores` equals \(\log p[\text{cand}]\); \(T=0\) ⇒ \(\deg\equiv 0\), so no value of \(\lambda\) changes
  the outcome; \(\lambda=1\) ⇒ \(\text{score}=-\deg\), picking the "least degenerate" candidate, yet **the candidate
  set never depends on \(\lambda\)** (changing λ does not change the set).
- Division of labour with neighbouring problems (details in `notes.md`): `top-k-sampling` also takes the
  \(\top_k\), but renormalizes inside the nucleus and then **draws a lottery**, with probability as its only
  criterion; `penalty-sampling` rewrites logits by **n-gram counts** from the history and does not pick a token;
  `mirostat-v2-sampling` adapts a truncation with a cross-step perplexity state \(\mu\). This problem uses
  **embedding similarity** and an argmax inside the candidate set.
