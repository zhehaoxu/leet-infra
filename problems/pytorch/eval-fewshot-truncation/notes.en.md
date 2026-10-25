# In Depth: Few-shot Prompt Truncation under a Token Budget

## Origin
- Wang, Prasad, Weber, et al., *How does in-context learning work? A framework for understanding the successes and failures of LLMs* (arXiv:2201.11903), which factorises in-context learning into **demonstration selection / ordering / formatting** and prescribes exactly this truncation rule for the budget case: keep demonstrations in their fixed order, and the moment the next one does not fit, drop everything after it — no re-selection, no re-ordering.
- Engineering: EleutherAI `lm-evaluation-harness` assembles the template as `fewshot + context + continuation` and fails with `context length exceeded` when it overflows; OpenCompass truncates by `max_num_tokens`; most evaluation scripts carry a `--max-shots` knob plus a tokenizer call. Our rule reproduces the "prefix of shots that fits" behaviour.
- Difference from `retrieval-rag-chunk`: that one splits **inside one document** into fixed-length overlapping chunks; here we choose among **whole demonstrations** in a fixed order and never touch their content.

## Formalisation
Let \(\tau(s)\) be the toy tokenizer's token count, \(w_i = \tau(\text{demo}_i)\) and \(b = \tau(\text{query}) + \tau(\text{tail})\). Then
\[ m = \max\Big\{\,m' \in \{0,\dots,M\} : b + \sum_{i < m'} w_i \le budget\,\Big\}, \qquad \text{kept} = [0, \dots, m-1], \qquad \text{final\_len} = b + \sum_{i<m} w_i . \]
Two structural facts are worth stating:
1. **The answer is a prefix**: since every \(w_i \ge 0\), whether demonstration \(i\) is accepted depends only on the accumulated cost of the previous ones, so the kept set can never have a hole. (It also means one could binary-search the prefix sums; the plain loop is what real harnesses write.)
2. **The reserve is the point**: subtracting \(b\) up front is precisely "the query and the label tail may not be spent on". Real evaluation code writes it as `budget - len(tokenize(template))`; forgetting it is what makes the tail fall off the end of the prompt, and then the model never sees `"Answer:"` — a silent failure that also breaks the answer-extraction regex downstream.

## The toy tokenizer and its traps
`tokenize` splits on whitespace; a word present in `VOCAB` is 1 token, a word absent from it is expanded **character by character** (each character an `<unk>`, id 0).
- Case sensitivity is real: `here` is in the table (1 token) while `HERE` is not (4 tokens). In the worked example `the dog is very nice` costs 8 rather than 5 exactly because `very` explodes into four character tokens.
- Because segmentation only looks at whitespace, \(\tau(a \Vert b) = \tau(a) + \tau(b)\) for a single-space join, so "accumulate per demonstration" and "tokenize the assembled string once" **coincide** and `final_len` is unambiguous. The test suite asserts the assembled-prompt length directly.
- **This stops being true for a real BPE tokenizer**: byte-level BPE merges and splits across the concatenation boundary, so \(\tau(a)+\tau(b)\) differs from \(\tau(a\Vert b)\) by one or several tokens. Production code must either encode the whole assembled string or use the chat template's `add_special_tokens` accounting; estimating the budget by summing part lengths is precisely the approximation that this toy setting hides. Keep that in mind before porting the numbers.

## Core points and pitfalls
1. **Hard cut, not a knapsack**: when demonstration \(i\) does not fit, `break`. Writing "skip it and look at the next, shorter one" (or sorting by length and packing) produces non-prefix indices — the classic mistake, and the test with a long demonstration followed by 1-token ones exists to catch it.
2. **The boundary is \(\le\)**: filling the budget exactly must keep that demonstration; using `<` silently loses a shot and makes the shot count wobble with prompt wording.
3. **Empty set and zero cost**: `demos = []` returns an empty tensor; a whitespace-only demonstration costs 0 and must be acceptable (since \(+0 \le budget\)). Do not "optimise" by skipping zero-cost entries.
4. **Report over-budget honestly**: if \(b > budget\) then `final_len > budget`. That is not an error but the truthful signal "the template alone already does not fit"; clipping the length to `budget` would fool every downstream length check.
5. **Return types**: indices are integers, so an int64 tensor (empty case must still be `torch.long`, not float); the length is a Python int, not a tensor.
6. **Order is semantics**: demonstration order measurably changes in-context performance (recency anchoring, majority-label bias), so "fixed order plus prefix truncation" is part of the metric definition; reordering to fit one more shot changes what is being measured.

## Verification approach
The reference solution reserves \(b\), adds `len(tokenize(demo))` demonstration by demonstration and `break`s at the first overflow. The judge's `_oracle` takes a different route: its own character-by-character whitespace scanner (no `str.split`) counts tokens, and the main loop is a `while` with an explicit overflow check returning a plain list. Assertions cover the worked example, a generous budget (everything kept), budget 2 (nothing kept), the exact \(\le\) boundary, the no-skip-ahead rule, the tail surviving even when it busts the budget, the empty demonstration set, the OOV character explosion, and re-tokenizing the actually assembled prompt to confirm `final_len`; plus 25 randomised demonstration/budget combinations checked against the oracle. Comparisons are exact (`torch.equal` / `==`) since no floating point is involved.

## Complexity
One linear scan per demonstration: \(O\!\left(\sum_i |\text{demo}_i| + |query| + |tail|\right)\) time and \(O(m)\) output space — the same order as the prefix-sum plus binary-search formulation.

## Practical notes
A real harness also reserves `max_new_tokens` and the template's special tokens from the budget, and the shot count is itself a hyper-parameter: truncating to 5 versus 8 shots can move the same model on the same data set by several points, so any reported number must state the budget and the truncation policy.
