# notes (ag-toolformer-filter-ledger)

## Sourcing

- **Toolformer has no official code release.** Authoritative sources: ① arXiv 2302.04761 **v5** (NeurIPS 2023) Sec 2.2 Filtering / Sec 4.1 weighting / App A thresholds — note v1 and v5 agree (both use L_i⁺/L_i⁻ weighted CE); the community-circulated three-condition P_plus/P_minus/P_corr formulation (τ+δ thresholds, 3 corrupted samples) **does not exist in either ar5iv version** — the todos candidate note ("perplexity gate, self-consistency dedup, weighted perplexity gain") was a mis-recall, corrected to the L formulation per v5. The ar5iv ?v1 suffix appears to serve the latest version anyway; version checks need multiple sources. ② lucidrains/toolformer-pytorch's filter_tokens_with_api_response / replace_all_but_first (the most-starred reproduction) — it implements loss⁻=min(no-response, no-call), τ=1.0, dedup keeping the first.
- Paper defaults: τ_s=0.05, τ_f=1.0, k=5, m=5; calculator/MT use τ_s=0, k=20, m=10, τ_f=0.5 (App A). Weights w_t = w̃_t / Σ w̃_s with Σ=3.0; lucidrains' comment admits "not sure what w_s is in the denominator" — this problem pins the global-constant-3.0 reading.
- The paper's filtering section has no dedup; "at most one API call per input" is an **inference-time** limit (Sec 4.2). Text-level dedup follows lucidrains' replace_all_but_first (training-data pipeline).

## Key quirks (verified)

1. **Two comparison operators point opposite ways**: the sampling gate `p_i > τ_s` is **strict**; the filter gate `L⁻ − L⁺ ≥ τ_f` is **inclusive** (verbatim "only keep API calls for which L_i⁻ − L⁺ ≥ τ_f holds").
2. **Global weight normalization**: Σ_{s∈ℕ} max(0,1−0.2s) = 3.0 is a constant — short suffixes are NOT re-normalized (2-token all -1 → L=0.6); t≥5 weights are 0 but the lp list still must carry those slots (extreme logps don't affect L).
3. **L⁻ min of two controls**: L(ε) (no call) vs L(e(c,ε)) (call without response) — makes both "call harmless but response useful" and "response useless but call text useful" comparable.
4. **lucidrains dedup regex, three quirks**: the leading space `' ['` is part of the pattern (calls at line start don't match); `\w+` tool name (empty names don't match); args `[^)]*` truncated at `)` (args containing parens don't match at all). Removal is **replacement with ''** (not deleting separators).
5. has_calls gate: the data pipeline drops call-less pairs entirely (the else branch of filter_and_keep_only_first_api_call).

## Engineering

- **The float exact-boundary gate is the biggest trap here**: an all -1 suffix gives score = 0.9999999999999999 < 1.0 — took three rounds trying to build a "score exactly equals tau_f" anchor to catch the `>=`→`>` mutation through the `(w̃/3)·lp` rounding. Fix: route the inclusive-gate evidence through an exactly representable construction — **identical tables → L⁺=L⁻ → score exactly 0.0 with tau_f=0** (0.0>=0 True; 0.0>0 False). The float quirk itself became a pinned ledger point (dropped at τ_f=1.0). Lesson: float exact-boundary anchors need exactly-representable constructions (even 0.2*5=1.0 exact breaks once you divide before multiplying: (0.2/3)*5 ≠ 1.0).
- **lp-list alignment ambiguity**: v1 wrote the weight step as t = j − pos (absolute index), contradicting the suffix-aligned test constructions — unified to **suffix alignment** (t = lp list index; pos is identity + per_pos grouping only).
- Validation must be **upfront over all candidates**: v1 only checked inside the eval loop, so a pos=9 candidate that failed sampling skipped validation (caught by the ValueError anchor).
- scores output order = evaluated order (positions ascending, input order within a group), not candidate input order — v1 used input order, caught by random.
- Mutations: 20 substantive all CAUGHT + 3 equivalences recorded: eq-min-commute (min commutativity), eq-order-tuple (top-k tie-break by passed index t vs position id i — the order-preserving filter makes t↔i strictly increasing), eq-kept-sorted (sort-then-map vs map-then-sort — equivalent for the same reason).
- regex-name-plus-star (\w+→\w*) SURVIVED in round one because the empty-name anchor string was wrong (' [](2)' is empty brackets followed by (2); the correct empty-name form is ' [(2)]') — lesson: print the counterexample character by character before building it.
- Red lines: real LM forward/logits, tokenizers, API executors, and the training loop are not modeled; the inference-time decoding limit is noted, not included.

## Follow-up candidates

- The sampling-side ledger (generating the p_i sequence) needs a real LM forward — only viable with logits as explicit inputs (softmax normalization + <API> token probability ledger).
- ToolLLM DFS decision tree (2307.16789) is the next problem; shares the "candidate filtering" context but the mechanism is independent.
