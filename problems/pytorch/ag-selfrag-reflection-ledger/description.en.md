# Self-RAG Reflection Token Scoring Ledger (ag-selfrag-reflection-ledger)

**Difficulty**: Hard | **Collection**: agentic-systems | **order**: 729

Stateless-ify Self-RAG's (arXiv **2310.11511**) inference-time "adaptive retrieval gate + evidence reranking" machinery: given the reflection-token log-probs of the no-retrieval generation, each evidence's generated tokens with per-position log-probs, and the scoring switches, recompute the `do_retrieve` decision, the segment-level relevance/ground/utility scores and the final answer choice. Source: [AkariAsai/self-rag](https://github.com/AkariAsai/self-rag) main — `retrieval_lm/run_short_form.py::call_model_rerank_w_scores_batch`, `postprocess_answer_option_conditioned`, and the reflection-token name lists plus `postprocess` in `retrieval_lm/utils.py`.

## What to implement

`solve(no_retr, evidences, second_text, mode="adaptive_retrieval", threshold=None, w_rel=1.0, w_sup=1.0, w_use=0.5, use_seqscore=False, use_groundness=False, use_utility=False, closed=False)`: return the retrieval-decision ledger, the per-evidence score ledger and the answer-aggregation ledger.

- `no_retr`: the no-retrieval generation `{"text", "lp0"}` (`lp0` is the `{token: logprob}` map at generation **position 0**).
- `evidences`: `[{"text", "tokens", "lps", "cumulative_logprob"}]`, one retrieval-augmented generation per evidence; `tokens` and `lps` have equal length.
- `second_text`: the text of the **second** generation (prompt with `"[No Retrieval]"` appended) when `do_retrieve` is False.

## Ledger points (line-by-line source check)

1. **The gate divides raw log-probs, not probabilities**: `do_retrieve = lp[Retrieval] / (lp[Retrieval] + lp[No Retrieval]) > threshold`. Log-probs are negative, so dividing negatives **flips the direction**: the less the model wants to retrieve (smaller `lp[Retrieval]`), the closer the ratio gets to 1 and the more likely it triggers — e.g. with `lp[No Retrieval] = -0.1` and `lp[Retrieval] = -2.0` the ratio is 0.952 > 0.5 and it retrieves, while the exp-ed probability ratio is only 0.11. The test is **strictly greater** (exactly 0.5 does not trigger).
2. **Two short-circuits and one degeneration**: `mode="always_retrieve"` is always true, `"no_retrieval"` always false (neither consults the gate nor emits `retrieval_ratio`); `threshold=None` (the CLI default) degrades to a **string containment test** `"[Retrieval]" in text`; the argparse default `"default"` is not in `choices` and falls through to the adaptive branch.
3. **`[Continue to Use Evidence]` is scored but never enters the gate**: `ret_tokens` holds three tokens, the ratio uses two.
4. **Two different missing-token fallbacks**: in the gate, `if id not in logprobs: score[tok] = -100` is **dead code** (the next line indexes unconditionally, so a missing key raises KeyError immediately); relevance/ground/utility use `exp(float(prob))` with a missing-token fallback of `exp(-100)` ≈ 3.7e-44 (a **subnormal, not 0**) — when both relevance tokens are missing the normalized score is exactly 0.5.
5. **relevance**: `P([Relevant]) / (P([Relevant]) + P([Irrelevant]))`.
6. **ground**: read log-probs only at the **first** position where a ground token appears (`break`), then `P(Fully)/Σ + 0.5·P(Partially)/Σ` (the third token carries weight 0); 0.0 (not 0.5) when the token never appears.
7. **utility**: likewise at the first `[Utility:k]` position, with the weight table **[-1, -0.5, 0, 0.5, 1]** — preferring `[Utility:1]` yields a negative score. The official position scan does **not** break (it collects all then takes the first, equivalent).
8. **final**: `w_rel·rel + w_sup·ground + w_use·utility` (defaults 1.0/1.0/**0.5**), plus `exp(cumulative_logprob / max(len(tokens), 1))` when `use_seqscore=True`. `use_groundness`/`use_utility` default to **off** (matching the CLI flags), so even a present reflection token scores nothing.
9. **Two aggregation flavours**: when `len(results) == 1` it returns `postprocess(pred)` directly (`pred` is the second generation while `results["no_retrieval"]` stores the first); otherwise `closed=True` accumulates scores **by answer text** and takes the max (identical texts merged, ties keep first appearance), while `closed=False` takes the highest-scoring path and returns the **raw, un-postprocessed** text.
10. **The no-retrieval result never joins aggregation**: `results["no_retrieval"]` has no score and both aggregation paths skip it by key — so with `do_retrieve=True` the first no-retrieval answer is discarded.
11. **The two postprocessors are asymmetric**: `postprocess` strips the 14 control tokens plus `</s>` and then removes **one** leading space; `postprocess_answer_option_conditioned` strips the control tokens plus `</s>`, `\n` and `<|endoftext|>` but does **not** touch the leading space.
12. **Domain constraints (ValueError)**: `lp0` missing a reflection token on the `threshold` path (official KeyError); `tokens` and `lps` of unequal length or empty (official `lps[0]` out of range); `do_retrieve` true with no evidence (official UnboundLocalError / IndexError on the empty aggregation).

## Example

```python
lp0 = {"[No Retrieval]": -0.5, "[Retrieval]": -1.5,
       "[Continue to Use Evidence]": -2.0}
ev = [{"text": "paris [Relevant]", "tokens": ["paris"],
       "lps": [{"[Relevant]": -0.357, "[Irrelevant]": -1.204}],
       "cumulative_logprob": -1.0},
      {"text": "lyon [Relevant]", "tokens": ["lyon"],
       "lps": [{"[Relevant]": -1.609, "[Irrelevant]": -0.223}],
       "cumulative_logprob": -1.0}]
led = solve({"text": "answer? [No Retrieval]", "lp0": lp0}, ev, "no",
            threshold=0.5)
# gate: -1.5 / (-1.5 + -0.5) = 0.75 > 0.5 -> retrieve
# best_path == "retrieval_0"; answer == "paris [Relevant]" (raw, uncleaned)
```

## Scoring

12 check points: the inverted gate with its strict inequality, the string-degenerated gate and the mode short-circuits, the second generation with single-space `postprocess`, relevance normalization with the `exp(-100)` subnormal fallback, ground's first-position scan and 0.5 weight, utility's negative weights, seqscore length normalization, the default switches, both aggregation flavours (text merging, stable ties), the single-result branch, domain constraints, plus a 120-case random reflection-table cross-check.
