# Context Compression: LongLLMLingua Document Budget, Dynamic Ratios and Window Threshold Plan

## Problem

LongLLMLingua (2310.06839) adds three layers on top of LLMLingua: rank documents by "question-aware PPL" and pick them under a token budget (document level), assign each selected document a different keep ratio (relevant ones keep more), then delete tokens inside a 200-token sliding window by a per-token NLL threshold (token level). In the real implementation each layer hides one easily-dropped detail: the sort direction depends on the condition type, the budget expression is a string arithmetic, and window boundaries are booked with `len+2` per document rather than the raw token count. This problem takes document PPL and per-token NLL as explicit inputs and reproduces the deterministic ledgers of all three layers.

## Task

Implement `solve(doc_token_ids, doc_ppl, token_loss, target_token, context_budget="+100", force_context_ids=None, reorder_context="original", dynamic_context_compression_ratio=0.0, condition_in_question="none", iterative_size=200)`:

- `doc_token_ids`: $B\ge 1$ documents, each a non-empty `list[int]`;
- `doc_ppl`: `list[float]` of length $B$, the **question-aware PPL score** per document (computed by `get_condition_ppl` upstream, given explicitly here; upstream multiplies the length term by `2/250*0`, i.e. length does not enter the score);
- `token_loss`: per-token NLL over the concatenated stream (upstream aligns `loss` to next-token prediction; this problem aligns to the token itself); `10000.0` is a sentinel that only leaves the threshold statistics and never affects keep decisions;
- `target_token`: compression budget (real number); apply `target_token < 0 → 100` first, then `context_budget`;
- `context_budget`: a string like `"+100"` or `"*0.5"` (`+ - * /` followed by an unsigned number) applied to the budget once;
- `force_context_ids`: document indices kept unconditionally (they consume budget too);
- `reorder_context`: `"original"` (restore ascending document order) or `"two_stage"` (even positions of the selection order first, odd positions reversed after);
- `dynamic_context_compression_ratio`: $\lambda \ge 0$ (upstream takes `abs`);
- `condition_in_question`: `"none"` sorts by PPL **descending** (without a question, high-information documents enter the budget first), `"question"` **ascending** (lower conditional PPL = more relevant); the sort is stable;
- `iterative_size`: window width $W \ge 1$.

Return a `dict` with keys `rank_order`, `selected`, `dynamic_ratios`, `tau`, `window_plan`, `thresholds`, `keep_masks`.

## Rules

**① Ranking.** `rank_order` = stable sort by `direction * ppl` with `direction = -1` (`none`) or `+1` (`question`).

**② Budget and selection.** Apply `target<0 → 100` and the arithmetic to get the **budget value** (called `budget` below; the window stage uses exactly it — the decrements made while selecting only decide the stopping condition). Walk `rank_order` subtracting each document's **raw token count**: `budget -= len(doc)`, add the index to the selection (dedup), stop once `budget < 0`. `force_context_ids` seeds the selection. Then apply `reorder_context` to get the output order `selected`.

**③ Dynamic ratios.** With $N$ = selection size (pre-reorder, in ranked order) the ratio table is

$$\Big[i \cdot \tfrac{\lambda}{N-1}\Big]_{i=-(N-1),\,-(N-3),\,\dots,\,N-1}\ \text{then fully reversed,}$$

0 when $N=1$; map it by document index and **follow the documents** into the `selected` order (not the positions). All zeros when $\lambda = 0$.

**④ Window plan.** In output order each document length is $L_i = \text{len} + 2$ (upstream's CLS/SEP allowance; document selection uses the raw length without +2 — the mismatch is part of the exam). $\tau = \dfrac{budget}{\sum_i L_i + 1}$ and the keep ratio $r_i = \mathrm{clamp}(\tau + \text{dynamic\_ratios}_i,\ 0,\ 1)$. Pack documents into windows starting from a virtual BOS (`last = 1`): when `last + L >= W` close the window with `delta_end = W - last` and set `last = last + L - W`; if the remainder exceeds $W$, factor out $k = \lfloor last/W \rfloor$ whole `(W, r)` windows and keep the remainder as the next window's head (empty remainder leaves nothing); otherwise `last += L`. A trailing segment closes normally.

**⑤ Thresholds and keeps.** The $g$-th flattened pair (upstream advances `end` by $W$ per pair) covers block $[gW,\ gW + \text{delta\_end})$. Its threshold: `ratio == 1.0` gives `-inf` directly; otherwise sort the values of `token_loss` that are `!= 10000.0` **descending** and take index `max(0, min(len-1, int(len*ratio) - 1))`. Keep mask: inside the block `loss > threshold` (position 0 always kept), outside always kept. Note `ratio` is the **keep** ratio — high-NLL (informative) tokens survive, and `r=0` empties the block.

## Example

```text
Example 1 (budget + two_stage + dynamic ratios + window plan)
doc_token_ids = [[10,11,12], [20,21], [30,31,32,33]]   # raw lengths 3, 2, 4
doc_ppl = [2.0, 8.0, 5.0], condition = "none" → rank_order = [1, 2, 0]
target = 10, budget "+5" → 15
  subtract in order: doc1 15-2=13 → doc2 13-4=9 → doc0 9-3=6 (never negative, all kept)
  used (ranked) = [1,2,0]; two_stage: even positions [1,0] + reversed odd [2] → selected = [1,0,2]
λ = 0.6, N=3: table [0.6, 0.0, -0.6], mapped by index → dynamic_ratios = [0.6, -0.6, 0.0]
Window (iterative_size=8): L = [4, 5, 6], budget=15 → tau = 15/(15+1) = 0.9375
  r = [1.0, 0.3375, 0.9375]
  last=1: doc1(4) packed; doc0(5): 5+5>=8 → close (3, 0.3375), remainder 2; doc2(6): 2+6>=8 → close (6, 0.9375), remainder 0
  window_plan = [[[4,1.0],[3,0.3375]], [[2,0.3375],[6,0.9375]]]

Example 2 (threshold and keep; token_loss has no sentinel, 9 tokens)
loss = [3.0, 1.0, 4.0, 1.5, 3.5, 0.5, 2.5, 2.0, 2.8]
pair (3, 0.3375) (block 1, block [8,11) truncated to [8,9)):
  int(9*0.3375)=3 → descending index 2 = 3.0; position 8 has loss=2.8 ≤ 3.0 → dropped
  keep_masks[1] = [T,T,T,T,T,T,T,T,F]
pair (4, 1.0): ratio=1.0 → -inf, all kept
```

## Constraints

- `doc_token_ids` non-empty, every document non-empty; `doc_ppl` matches the document count with finite real values.
- `len(token_loss)` equals the total token count; `10000.0` is the sentinel.
- `context_budget` must match `^[+\-*/]\d+(\.\d+)?$`; `reorder_context` and `condition_in_question` accept only the listed values.
- `force_context_ids` indices must lie in `[0, B)`; `iterative_size` is an `int >= 1`.
- Inputs must not be modified; all invalid inputs raise `ValueError`.

## Boundaries with Neighboring Problems

- `ctx-llmlingua2-word-compress` is LLMLingua-2's **word-level classification post-processing** (BERT scores, no document structure); this problem is LongLLMLingua's **document budget + window threshold plan** (LLaMA scores, document PPL given).
- `distributed-cp-zigzag-partition` / `distributed-cp-varlen-zigzag-plan` are CP's **static layouts** (causal load balancing); this problem is the **dynamic allocation** of a compression budget (by PPL and window NLL).
