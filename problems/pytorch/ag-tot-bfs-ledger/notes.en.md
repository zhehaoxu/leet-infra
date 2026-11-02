# notes — ag-tot-bfs-ledger (order 720)

## Source of truth

- Official repo princeton-nlp/tree-of-thought-llm; **code lives under `src/tot/`** on master (the README's `tot/methods/bfs.py` path is stale, raw fetch 404s).
- The `solve` loop: start `ys = ['']`; each step generate → score → select → `infos.append({step,x,ys,new_ys,values,select_new_ys})` → `ys = select_new_ys`. This problem mirrors that structure with the `value` evaluation branch and the `greedy` selection branch.
- game24 `value_outputs_unwrap`: `if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower(): return 0`; `value_names = [_.split('\n')[-1] for _ in value_outputs]`; `value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}` (**officially annotated "TODO: ad hoc"**); `value = sum(value * value_names.count(name) ...)` — **no division by len(value_outputs)**.
- `get_values` local_value_cache: `if y in local_value_cache: value = 0` — a repeated candidate is hard-zeroed **instead of reading the cached value**, and the cache is written only on first computation.

## Key ledger points

1. **Duplicates score 0**: one parent proposing the same line twice (e.g. output "x\nx") → the second candidate is 0, its eval texts ignored. Both mutations `value = 0 → value = local_value_cache[y]` and `if y in cache → if False` are caught by this anchor.
2. **No normalization**: two "sure" lines score 40.0; the `/ len(eval_outputs)` mutation is caught twice — by the empty-eval-list degenerate case and the 40 anchor.
3. **Only the last line counts**: `sure\nbecause of x` has a non-key last line → 0; case-sensitive (`SURE` does not match).
4. **Four-line-no-answer penalty**: counts lines after strip (propose children always carry a trailing `\n` — `"a\nb\nc\n".strip()` is 3 lines, no trigger; sample texts can carry newlines and become 4/5 lines at once). `== 4` vs `>= 4` requires a 5-line anchor; the case-insensitive `answer` exemption vs raw matching is separated by the `Answer: x` anchor.
5. **Concatenation difference**: propose `y + p + '\n'` vs sample `y + s` — mutations (drop/add newline) caught by exact new_ys assertions plus cross-step 4-line behavior.
6. **Greedy stable sort**: `sorted(ids, key=value, reverse=True)` keeps original index order on ties; the `key=(value, id)` mutation flipping tie order is caught.
7. **propose adds one line per step**: building a 4-line y needs a 3-step propose chain or a single sample text — anchors cover both.

## Not modeled (recorded)

- `task.value_cache` (cross-step, prompt-keyed): depends on real prompt construction, excluded; the per-step local duplicate cache is included.
- `method_evaluate='vote'` (`get_votes`) and `method_select='sample'` (`np.random.choice` defaults to **with replacement** + global RNG): randomness red line, excluded.
- Unknown `method_generate` has **no else branch** officially → `NameError: new_ys`; this problem raises `ValueError`, divergence recorded.
- game24's `propose_prompt_wrap`/`get_current_numbers` (`left: ` parsing) is prompt-side; eval texts are inputs, excluded.

## Authoring lessons

- **The oracle's float summation order must match the official**: official `sum()` folds in weight-table order (imp→likely→sure); the first oracle accumulated in eval-text order, and the 1e-15 gap between `40.001` and `40.001000000000005` flipped the order of two "equal" candidates (caught at random iteration 129). Fixed by counting in official order.
- **evals[i] is per child** (each entry being that child's list of eval texts), not per eval text — mistyped twice while hand-writing anchors.
- **propose adds exactly one line per step**: a 4-line y needs a multi-step chain or sample texts — count actual line numbers before designing anchors.
- 3 equivalent mutations (redundant `list(...)` copies in infos for `ys/new_ys/select_new_ys`; none of these lists is mutated in place) confirmed SURVIVED, recorded to avoid future confusion.
- All 18 substantive mutations CAUGHT.
