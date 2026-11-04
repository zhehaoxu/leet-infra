# Solution Notes

## Approach

The deterministic core of the official STaR repo (`device_inference.py`):

1. **eval_output** (L27-91): per-sample extraction chain (cqa first-line `[-3]` char / gsm first `####` line truncation / arithmetic line-list index chain), `lower()` comparison, training-example assembly (base + extraction intermediate + endoftext), and per-class `{accurate, total}` counting;
2. **get_score** (L94-97): `total==0 -> 0` (int);
3. **question_to_context** (L99-124): context assembly for the three datasets (cqa choice list with (CORRECT) marks / gsm answer hint / arithmetic adjacent `Target:` split pairs);
4. **eval_examples** (L207-238) + the tokenization-free part of **examples_to_batch** (L127-186): batched evaluation, missed samples admitted to a hint pool with `p_rationalization`, pool flushed as a `key+'_r'`-renamed batch, `p_show_hint_save` swapping the saved context to the hinted version with some probability, `few_shot_train` adding a few-shot save prefix (arithmetic raises).

Out of judging scope: mesh_transformer/jax `network.generate` and numpy batch padding (outputs are explicit decoded strings), GPT2 tokenizer encode/decode (contexts are consumed by the `gen_fn` stub), wandb/tqdm/checkpoint/smart_open, and `iteration_train.py`'s os.system orchestration (gen_train/gen_records/train_model/eval_model) plus the `get_n_steps` growth accounts (steady/exponential/count-data-points — pure shell orchestration with no incremental branching semantics, not included).

Boundaries with existing problems:
- **775/776/777/778/779**: demo assembly, candidate scoring, RoT sequence compilation, CR state graph; this problem is an **evaluate-writeback-rationalize** loop — the ledger's core is the two flows "assemble correct samples back into the training set" and "re-evaluate missed samples under hint contexts";
- **779's three-verdict verifier**: CR's verdicts drive graph growth; the hint pool here is a stateless FIFO (`_r` renaming + carry-over), and correct samples are written out directly without entering any graph.

## Implementation details (pinned to the official code)

- **total increments before the try**: `accuracy[example_class]['total'] += 1` sits before `try:` — short outputs skipped by IndexError still count. `get_score` returns **int 0** (not float) when total==0.
- **cqa extraction**: `output_numbers[0]` takes only the first line -> clip the endoftext substring -> `[-3]` **character** index; len<3 raises IndexError -> `except IndexError: pass` silently skips. Later lines of multi-line outputs never participate.
- **gsm extraction**: scan the **first** line containing `'####'`, `output_numbers = '\n'.join(lines[:line_idx+1])` truncates (the saved example uses this truncated, **unstripped** string), clip endoftext, prediction `split('####')[-1].strip()`; without `####` the prediction is `""` and never saves (output_numbers stays a list; if target were `""` and it "hit", concatenating would raise TypeError — the official code only catches IndexError; this task keeps the same semantics).
- **arithmetic extraction**: `output_numbers` is still a **list of lines** — `'<|endoftext|>' in output_numbers` and `'</scratch>' in output_numbers` are **whole-line equality** membership tests (not substring checks); prediction-line index chain: endoftext line `-1` (line 0 gives `-1`, Python negative indexing takes the **last line**), `</scratch>` line `+1` (==len skips via continue), otherwise `direct and len>1 -> 1 else 0`; the save `'\n'.join(lines[:pi+1])` is then **clipped once more** for endoftext substrings (cross-line tokens only get removed here).
- **Correctness**: `output_prediction.lower() == target.lower()` on both sides; writeback = `base_context + extraction intermediate + endoftext`.
- **eval batch loop**: evaluate only when `len(main_examples) == eval_batch_size` (a **short trailing batch is never evaluated**); after each batch, each miss flips `rng.random() < p_rationalization` (**short-circuit `and` — the coin is only flipped for misses**) into the pool; `rationalize and len(pool) >= eval_batch_size` flushes the **first** batch immediately (renamed `key+'_r'` -> separate accuracy bucket) with `hint_prompts, hint=True`, and `hint_examples = hint_examples[batch:]` keeps the remainder (carry-over: pool 4, batch 3 -> flush 3, keep 1).
- **Per-batch contexts**: hintless and base (hinted) computed separately; for arithmetic `cur_few_shot = base.split('\n\n')[:-1]` **overwrites** the passed few-shot (recomputed per sample); both take `split('\n\n')[-1]`; `rng.random() < p_show_hint_save` swaps the saved hintless context to base (observable only in hint batches — with hint=False the two are identical); `few_shot_train` prepends `'\n\n'.join(train_prompts) + '\n\n'` to the saved context (arithmetic raises NotImplementedError right at batch preparation, before any correctness check).
- **Target extraction** (examples_to_batch): arithmetic takes `split('\n')[-3]` ('invalid' if <3 lines), cqa takes `answerKey`, gsm takes `answer.split('#### ')[-1]` (with the trailing space — different from eval_output's `'####'`).
- **Coin order**: per-sample `p_show_hint_save` flips within a batch come before the per-miss `p_rationalization` flips after the batch; hint batches flip in the same order. The rng is an explicit `random.Random` parameter.

## Test design (22 cases)

extract 6 (cqa including a multi-line hit / endoftext clip / IndexError; gsm including no-`####`, double-`####`, strip, case-insensitivity; arithmetic six paths including the `-1` negative index and whole-line membership; direct single/multi-line; few_shot_train raise; length mismatch), score 1 (int 0 and 0.375), context 3 (cqa including uppercase answerKey matching, gsm hint, arithmetic four variants including direct+hint and no-Target), eval 11 (leftover never evaluated, short pool no flush, `p_show_hint_save=1.0` landing (CORRECT) on disk, p_rat=0, few_shot_train prefix + arithmetic raise, arithmetic few-shot overwrite, `_r` bucketing, double flush, p_rat=0.5 coin sequence (Random(7), measured in the official sandbox), carry-over pool-4 flush-3 keep-1, rationalize=False pool that never flushes), args_validation 1 (11 invalid-input paths). All expected values were pinned by exec-sandboxing the official source (stub args/basic_open/tqdm/transformers/random; eval_batch mirrored with a shim reproducing the tokenization-free part of examples_to_batch).

## Deviations from the official code

- Official `eval_output` `print(..., file=new_train_f)`s training examples to a file; this task returns a `train_examples` list instead (port-equivalent: file content = the list concatenated item by item).
- Official `eval_examples` uses `transformers.GPT2TokenizerFast` + `network.generate` + `tqdm`; this task replaces them with the `gen_fn` stub (outputs already decoded); the `pbar.set_description` score display is out of judging scope.
- Official `args.dataset_mode`/`args.few_shot_train`/`args.no_prompt` are module-level globals; this task passes them explicitly (`prompts=None` iff no_prompt).
- Official gsm with no `####` and an empty-string target that "hits" would raise TypeError when concatenating a list + str (uncatched officially); this task keeps the same semantics without special-casing.
