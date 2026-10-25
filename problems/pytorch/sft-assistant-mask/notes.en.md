# In Detail: SFT Assistant-Reply Loss Mask

## Origin
An engineering convention rather than a single paper: ChatML / Llama chat templates tag every turn with a role; TRL's `DataCollatorForCompletionOnlyLM` (`response_template="[/INST]"` etc.) and the SFT scripts of various projects all implement assistant-only supervision by "masking the prompt and keeping only the completion". Whether the run-final EOS is supervised depends on the training recipe (you normally supervise EOS to teach the model "when to stop", and can turn it off otherwise); this problem exposes it as the `supervise_eos` switch.

## Core Points and Pitfalls
1. **Run detection**: an assistant run is a **maximal** contiguous run of role equal to `assistant_id`, separated by role changes; the run tail is handled per run.
2. **EOS is dropped only at the run tail**: `supervise_eos=False` takes effect only at the **last** position of a run, and only if `token_ids[b,·]==eos_id`; an `eos_id` appearing in the middle of a run is part of the reply and is kept. This is the easiest place to get wrong — do not "truncate as soon as you see eos".
3. **Pads are excluded naturally**: the role at pad positions is not assistant, so they are automatically 0; no special-casing of pad_token_id is needed.
4. **Output values**: only 0.0 or 1.0, with shape strictly equal to the input \((B, L)\).

## Judging Protocol
The reference solution "finds the maximal assistant run row by row → decides whether to drop its tail → fills 1 inside the run". The judge's `_oracle` uses a **completely independent** position-wise rule: `mask[i]=1` if and only if `role[i]==assistant_id` and **not** ("i is the last position of an assistant run and token[i]==eos_id and not supervise_eos") — run-last is determined by "the next position's role≠assistant", a structure different from the reference's "find the run and step back one", but numerically equivalent. Additional assertions cover "comparing the two supervise_eos settings", "an eos inside a run is not dropped", "no assistant → all 0", and "multiple runs".

## Complexity
\(O(B L)\).
