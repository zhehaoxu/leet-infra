# Deep dive: Contrastive Search with Degeneration Penalty

## Origin

- **Yixuan Su, Tian Lan, Yan Wang, Dani Yogatama, Lingpeng Kong, Nigel Collier**,
  *A Contrastive Framework for Neural Text Generation*, **arXiv:2202.06417** (submitted 2022-02-13, v2 on
  2022-05-30). Contrastive search is the decoding method proposed in §3.2 of that paper (CNTRM is its label in
  the paper), paired with the SimCTG training objective. Checked against the source, the points that matter
  here are: ① at every step the output is selected "from the set of most probable candidates predicted by the
  model", the candidate set being top-\(k\), where the text says \(k\) is typically set to \(3\sim10\); ② in the
  selection formula (their Eq. 5) the first term is the **model confidence** \(\log p(v\mid x_{<t})\) and the
  second is the **degeneration penalty**, defined as "the **maximum cosine similarity** between the
  representation of the candidate and that of all tokens in" the context \(x_{<t}\) (the similarity function is
  their Eq. 3, cosine with range \([-1,1]\)); ③ \(\lambda\) weighs the two terms and "when \(\lambda=0\),
  contrastive search degenerates to the greedy search method". All three match this problem's interface and its
  \(\lambda=0\) identity.
- **Yixuan Su, Nigel Collier**, *Contrastive Search Is What You Need For Neural Text Generation*,
  **arXiv:2210.14140** (2022-10-25). The follow-up study: after evaluating isotropy across 16 languages it
  concludes that the anisotropy claim only holds for GPT-2-small/medium, while **the contrastive search decoding
  method itself works without any extra training** — which is exactly the practical justification for setting
  this problem without a language model, taking the similarity matrix as an input.
- **The two places where this problem departs from the paper (the problem statement / `starter.py` win)**:
  1. The paper uses the **raw cosine similarity** \(s\in[-1,1]\) as the penalty; this problem first maps it
     linearly to \(\deg=\frac{s+1}{2}\) and clamps to \([0,1]\). Since
     \(\max_i\frac{s_i+1}{2}=\frac{\max_i s_i+1}{2}\) (the map is monotone), the only differences are that
     "\(\lambda\)'s scale is off by a factor of 2" plus a **candidate-independent** shift \(-\lambda/2\): a common
     shift cannot change an argmax, so this problem's \(\lambda\) corresponds to \(\lambda/2\) under the paper's
     convention. The clamp to \([0,1]\) also preserves the order. The paper never writes \((1+s)/2\); this
     normalization is a **convention of this problem** (it keeps the penalty in \([0,1]\) and matches the
     intuition "similarity 1 = worst"). Verifying the normalization used by a specific upstream implementation
     is **pending source verification (to be traced back)**.
  2. The paper obtains each candidate's hidden state by running one more forward pass on "prefix + candidate";
     the grading environment has no language model, so this problem passes \(sim\in\mathbb{R}^{K\times T}\) in
     explicitly and stipulates that **the row order equals the descending-probability candidate order**.
- **Claims deliberately not adopted**: this collection's entry lists the source as "Chan et al., arXiv
  2202.10479, *Language Models with Contrastive Search*". Measured on 2026-10-25: the metadata of
  arXiv:2202.10479 is *A flavor-inspired radiative neutrino mass model* (neutrino physics, unrelated to text
  generation), and both the arXiv title search `ti:"Language Models with Contrastive Search"` and
  `au:"Chan" AND ti:"contrastive search"` return 0 hits. So **that identifier is not cited here**, and that
  sentence of the entry is marked **pending source verification**. Separately, the `Wu et al. 2022` note inside `solution.py` does not
  match the real authors of arXiv:2202.06417 (Su et al.) — a slip in the comment; the identifier itself is fine.
- **There is no off-the-shelf implementation to borrow in the grading environment**: `.venv-eval` ships
  `transformers` 5.20.2, where only the `GenerationMode.CONTRASTIVE_SEARCH` enum and one community-model URL
  survive; grepping `generation/logits_process.py` finds no contrastive / `penalty_weight` processor, and
  llama.cpp is absent. The scoring and the selection must both be written by hand.

## Full derivation and index conventions

Let \(z\in\mathbb{R}^{V}\), \(p=\operatorname{softmax}(z)\), \(\ell=\log\operatorname{softmax}(z)\), candidate
count \(K=\min(k,V)\), history length \(T\).

1. **Candidate order.** Sort the indices by the key \((-\,p_i,\ i)\) ascending and keep the first \(K\) to get
   \(\text{cand}_0,\dots,\text{cand}_{K-1}\). Softmax is monotone, so "descending probability" and "descending
   logit" select the same tokens; but **ties must be broken by ascending index**, otherwise two runs on the same
   input may swap the order. Note the two coordinate systems: "rank \(j\) inside the candidate set" versus
   "vocabulary index \(\text{cand}_j\)". `chosen` returns the latter, while position \(j\) of `deg` / `scores`
   belongs to the former — mixing them up is the classic source of out-of-range results.
2. **Degeneration penalty.** \(sim_{j,i}=\cos\big(h_{\text{cand}_j},\,h^{\text{hist}}_i\big)\).
   \[
   \deg_j=\operatorname{clamp}_{[0,1]}\Big(\max_{1\le i\le T}\tfrac{sim_{j,i}+1}{2}\Big),\qquad
   \deg_j\equiv 0\ \ (T=0).
   \]
   A \(\max\), not a mean: one history state that is nearly aligned with the candidate is enough to call it
   "degenerate". The ordering over \(i\) plays no role in the computation, so the **columns** of \(sim\)
   (history) may be permuted freely while the **rows** (candidates) may not. \(T=0\) is legal input (there is no
   history before the first token): pass a \(K\times 0\) empty tensor, the score collapses to
   \((1-\lambda)\ell\) and \(\lambda\) stops mattering at all.
3. **Score.**
   \[
   \text{score}_j=(1-\lambda)\,\ell_{\text{cand}_j}-\lambda\,\deg_j .
   \]
   The log uses the full-vocabulary log-softmax: it does not matter that \(\sum_i e^{\ell_i}\neq 1\). Do **not**
   renormalize inside the \(\top_k\) nucleus before taking the log — that changes the relative scale of the two
   terms, which is equivalent to silently swapping \(\lambda\).
4. **Argmax and ties.** \(j^{*}=\arg\max_j \text{score}_j\), ties resolved to the smallest \(j\). Implementation-wise
   this is a linear scan that updates only on a **strict** greater-than, matching the first hit of
   `torch.argmax` on CPU; writing `>=` instead jumps to the last tied candidate.
5. **The crossover point (checkable on the sample data).** Which of two candidates \(a,b\) wins is decided by
   \((1-\lambda)(\ell_a-\ell_b)\gtrless \lambda(\deg_a-\deg_b)\), i.e. the threshold
   \(\lambda^{*}=\frac{\ell_a-\ell_b}{(\ell_a-\ell_b)+(\deg_a-\deg_b)}\). In the sample, with \(a=0,b=1\):
   \(\ell_0-\ell_1=0.3\) and \(\deg_0-\deg_1=0.775\), hence \(\lambda^{*}=0.3/1.075=0.279070\).
   Measured: at \(\lambda=0.279069\) scores are \([-0.837489,\,-0.837490,\,-1.453769]\) and \(0\) wins; at
   \(\lambda=0.279071\) they are \([-0.837489,\,-0.837488,\,-1.453767]\) and \(1\) wins.
6. **Cross-step semantics (outside what this problem computes).** In real decoding the chosen token's hidden
   state joins the history, so candidates aligned with it see \(\deg\) pushed towards \(1\) and get suppressed —
   that is the "the more it repeats, the harsher the penalty" feedback loop. This problem only hands back the
   **single-step** 4-tuple, with the history played by the input \(sim\).

## Core ideas and pitfalls

1. **No random numbers.** This is the only deterministic decoding step in the collection: there is no \(u\), and
   inventing one is forbidden. Any `torch.rand` makes `chosen` differ between runs of the same input.
2. **The row order of \(sim\)**: row \(j\) belongs to the **\(j\)-th descending candidate**. Reading the first
   \(K\) rows in vocabulary order silently misaligns everything unless the candidates happen to be exactly
   \(0..K-1\). The sample has candidates \([0,1,2]\) and therefore cannot catch this bug; the randomized
   cross-check cases, whose candidate order is shuffled, do.
3. **The log is not nucleus-normalized**: use the full-vocabulary \(\log\operatorname{softmax}\). This is the
   opposite of `top-k-sampling` / `top-p-sampling`, where the kept set is renormalized before use; copying that
   habit here is wrong.
4. **Clamp after the max**: \(\operatorname{clamp}(\max)=\max(\operatorname{clamp})\) because clamping is
   monotone, so the two writings are equivalent. But clamping \(sim\) itself to \([0,1]\) (truncating even the
   *negative* similarities) instead of clamping \(\deg\) systematically raises every penalty and makes the result
   look greedy.
5. **Ties go to the earlier candidate**: on an exact `score` tie (the \(k>V\), \(sim=0\), two equal 1.0 logits
   line in the sample) rank 0 is chosen. When hand-writing the comparison instead of `argmax`, the test must be
   a strict `>`.
6. **Clamp \(k>V\)**: only after \(K=\min(k,V)\) is the number of \(sim\) rows defined; skipping the clamp
   indexes out of range.
7. **Two identities are hard checks**: \(\lambda=0\) ⇒ the outcome equals the global argmax and `scores` equals
   \(\ell[\text{cand}]\); \(T=0\) ⇒ \(\deg\equiv0\) and \(\lambda\) is irrelevant (the same token is chosen for
   every \(\lambda\)). \(\lambda=1\) ⇒ `scores` \(=-\deg\), the winner is \(\arg\min\deg\), and **the candidate
   set is unchanged**.
8. **dtype and ordering**: `cand_idx` must be `int64`; `deg`/`scores` keep the input dtype (float32) and must be
   emitted in exact candidate order, since grading compares the arrays elementwise.

## Differences from neighbouring problems

- **vs `top-k-sampling`**: both **do** start from the \(\top_k\), but there the \(k\) *is* the kept set — it is
  renormalized and then a **lottery** is drawn with the explicit \(u\), with probability as the sole criterion.
  Here the \(\top_k\) is only the **scoring range**, the selection score is dominated by the similarity penalty,
  and the choice is an argmax with no lottery. The same candidate set yields different tokens: in the sample the
  candidates \([0,1,2]\) produce token 1 at \(\lambda=0.6\), whereas top-k sampling would keep most of its mass
  on token 0.
- **vs `penalty-sampling`**: that one is **token-count driven** repetition suppression — repeat multiply/divide
  on logits by the occurrence count inside the `history` window, then subtract
  \(\text{count}\cdot\text{freq}+\text{present}\); it touches the **whole vocabulary**, happens **before** the
  softmax, and it **does not pick a token** (it returns penalized logits). This problem is **embedding-similarity
  driven**, touches only the \(K\) candidates, argmaxes inside the candidate set and returns `chosen` directly.
  The semantic-versus-literal gap is the sharpest one: a paraphrased echo survives `penalty-sampling` untouched,
  while here its \(\deg\) is large.
- **vs `dry-sampling`**: DRY finds **exact n-gram repeats** with a reversed Z-algorithm and penalizes logits by
  \(base^{R-\text{allowed}}\); it is likewise string-level and does not pick a token. This problem does no
  substring matching.
- **vs `mirostat-v2-sampling`**: Mirostat v2 keeps a cross-step state \(\mu\) (a target perplexity), truncates
  the prefix by \(-\log_2 p\le\mu\), draws a lottery, and hands back
  \(\mu_{\text{new}}=\mu-\eta(b-\tau)\) to the caller — it is a **feedback control loop**. This problem has no
  state, no feedback and no lottery; \(\lambda\) is a fixed hyper-parameter. Both chase "adapt to confidence
  while avoiding degeneration", but \(\mu\) travels the **entropy / perplexity** channel while this problem
  travels the **representation similarity** channel; replace \(\deg\) by entropy and you get the other family.
- **vs `top-n-sigma-sampling` / `min-p-sampling` / `top-p-sampling` / `typical-sampling`**: those are
  probability/statistics filters that are **independent of what has already been generated** (their thresholds
  come from the shape of the current distribution alone); here the score depends explicitly on the history
  states. Filters emit `filtered` plus a lottery result, whereas this problem emits candidates, penalties,
  scores and an argmax choice.
- **vs `xtc-sampling`** (same collection): XTC cuts the **probability head** (the contiguous prefix above a
  threshold) and never looks at history; this problem removes no token at all, only re-ranks inside the
  candidate set, and everything it knows comes from history.

## Cross-check methodology

The reference solution takes `softmax` + `argsort(descending=True, stable=True)` for the candidates, computes
`(sim+1)/2` then `max(dim=-1)` and `clamp(0,1)` for \(\deg\), and does one explicit strict-greater-than linear
scan for the argmax. The grading `_oracle` opens a second, independent path: pure Python `math.exp` plus
`math.fsum` for the softmax and the \(x-\max-\log(\text{sum})\) form of the full-vocabulary log-softmax,
`sorted(range(V), key=lambda i: (-p[i], i))` for the candidates, hand-written per-candidate per-history-column
maximum with clamping, list-comprehension scores, and the same strict `>` scan for the selection. The cases
cover 60 randomized \((V\in[3,13),\,k\in[1,V+2),\,T\in[0,5),\,\lambda\in\{0,0.35,0.6,1\})\) full cross-checks,
plus four dedicated cases: \(\lambda=0\) greedy, \(T=0\) zero penalty, \(\lambda=1\) least-redundant choice, and
\(k>V\) clamping with tied scores.

## Complexity

One softmax and one log-softmax, each \(O(V)\); taking the top-\(K\) costs \(O(V\log V)\) (down to
\(O(V+K\log K)\) with `torch.topk`); \(\deg\) costs \(O(KT)\); scoring and argmax cost \(O(K)\). Space is
\(O(V)\) for the probabilities and log-probabilities plus \(O(KT)\) for the similarities. In real decoding each
step needs one forward pass **per candidate** to get the hidden states, so the \(O(K)\) forward passes are the
true bottleneck; this problem folds that part into \(sim\), leaving only cheap tensor arithmetic.

## Practical notes

\(\lambda\) is tied to the scale of \(sim\): this problem puts \(\deg\) in \([0,1]\) while the paper keeps
\([-1,1]\), so the same numeric \(\lambda\) is 2× stronger under one convention than the other (the common shift
of \((1+s)/2\) does not affect the choice). The paper suggests \(k\) around \(3\sim10\) and devotes Appendix B to
ablating \(k\) and \(\lambda\) — **the recommended numeric ranges are pending source
verification**, this batch only verified the
main-text statement "top-\(k\), \(k\) typically \(3\sim10\)" and the existence of that appendix. An overly large
\(\lambda\) pushes generation towards candidates that are "unrelated but very dissimilar", which shows up as a
worse gen-ppl and artificially inflated diversity. A further engineering caveat: \(sim\) must use the hidden
state **after the candidate is appended to the prefix**; approximating it with the last hidden state of the
prefix alone makes every penalty too small and the decoder look greedy.
