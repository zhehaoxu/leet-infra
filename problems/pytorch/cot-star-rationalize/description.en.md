# STaR: Bootstrapping Rationalization Ledger (Three-Dataset Extraction + Correct-Writeback + Hint-Pool Rationalization)

## Problem

In STaR (arXiv:2203.14465, official repo `ezelikman/STaR`), `device_inference.py` organizes "generate -> extract answer -> judge correctness -> write correct samples back to the training set" into two deterministic ledgers: `eval_output` (L27-91) extracts answers per dataset, judges, assembles new training examples, and keeps per-class counts; `eval_examples` (L207-238) batches the dataset by `eval_batch_size`, sends wrong samples to a hint pool with probability `p_rationalization`, and once the pool holds a full batch, re-evaluates them under hint contexts with keys renamed `key+'_r'` (rationalization). This task makes all LLM generations explicit inputs (decoded `outputs` / a `gen_fn` stub) and tests the pure ledger:

```text
mode="extract": outputs + answers + base_contexts + classes + dataset_mode
                -> {"successful", "train_examples", "accuracy"}
mode="score":   counts {"accurate","total"} -> int 0 if total==0, else accurate/total
mode="context": example + dataset_mode (+hint/direct) -> context string
mode="eval":    dataset + prompts/hint_prompts + gen_fn + rng + eval_batch_size
                (+rationalize/p_rationalization/p_show_hint_save/few_shot_train)
                -> {"accuracy", "train_examples"}
```

## Ledger rules (pinned to the official code)

`mode="extract"` (official `eval_output`; **total increments before the extraction try**; an IndexError is swallowed by `except IndexError: pass` — the sample is counted but never saved):

```text
output_numbers = output.split('\n'); init accuracy[class] = {'accurate':0,'total':0}; total += 1
cqa:         first line -> clip '<|endoftext|>' substring -> prediction = line[-3] char
             (len<3 -> IndexError -> silently skipped)
gsm:         first line containing '####' -> truncate to that line (inclusive) -> clip eot -> split('####')[-1].strip(); no '####' -> empty prediction (never saved)
arithmetic:  output_numbers is still a LIST of lines — '<|endoftext|>' and '</scratch>'
             are **whole-line equality** membership tests:
               prediction line = endoftext index - 1 (index 0 gives -1 -> Python negative indexing takes the LAST line)
               '</scratch>' index + 1; if == len(output_numbers) -> skip the sample
               otherwise direct and len>1 -> index 1, else 0
correct:     clip endoftext from the prediction, compare with target via .lower() on both sides
save:        cqa/gsm: new_example = base_context + output_numbers + endoftext
             arithmetic: join(lines[:pred_line+1]) -> clip endoftext -> base + join + endoftext
             few_shot_train and arithmetic and correct -> NotImplementedError (official raise)
```

`mode="eval"` (official `eval_examples` + the tokenization-free part of `eval_batch`/`examples_to_batch`):

```text
append samples to main; a batch is evaluated when it reaches eval_batch_size (a short trailing batch is **never evaluated**);
after each batch, each missed sample with rng.random() < p_rationalization joins the hint pool;
rationalize and pool >= eval_batch_size -> immediately take the first eval_batch_size (key + '_r' -> separate accuracy bucket),
re-evaluate with hint_prompts and hint=True, keep the remainder in the pool.
Per sample within a batch:
  compute hintless (no hint) and base (hinted) contexts;
  arithmetic: few-shot is replaced by base.split('\n\n')[:-1], both contexts take split('\n\n')[-1]
  rng.random() < p_show_hint_save -> save with base instead of hintless
  few_shot_train: saved context gets a '\n\n'.join(train_prompts) + '\n\n' prefix
                  (arithmetic raises NotImplementedError immediately)
  input context = '\n\n'.join(few_shot) + '\n\n' (empty when prompts is None) + base
  target: arithmetic third-from-last line ('invalid' if <3 lines), cqa answerKey,
          gsm answer.split('#### ')[-1]
```

`mode="context"` (official `question_to_context`): three formats per the solution docstring. `mode="score"`: `total==0` returns **int 0**.

## Key quirks (official behavior)

- cqa's `[-3]` is a **character** index over the first line only, after clipping endoftext;
- gsm saves the **unstripped truncated string** (strip applies only to the prediction);
- arithmetic's endoftext/`</scratch>` tests are **whole-line equality** (mid-line substrings don't count), and endoftext on line 0 yields `-1`, i.e. the last line via negative indexing;
- arithmetic saves the join up to and including the prediction line, then clips endoftext substrings once more;
- the hint batch's class is `key+'_r'`, counted separately from the main batch; saves use the hintless context unless the `p_show_hint_save` coin fires;
- the `p_rationalization` coin is flipped **only for missed samples** (short-circuit `and`), after the per-sample `p_show_hint_save` coins within the batch.

## Input conventions

- `outputs`: decoded model output per sample; `answers`/`base_contexts`/`classes` aligned;
- `dataset`: list of `(key, example)` pairs; cqa/gsm examples are dicts, arithmetic examples are raw strings;
- `gen_fn(contexts, hint) -> list[str]`: deterministic generation stub;
- `rng`: a `random.Random` instance (coin order as above).
