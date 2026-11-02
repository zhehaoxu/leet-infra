# self-instruct: Synthetic Instruction Filtering and Similarity Dedup Ledger

## Problem

self-instruct bootstraps large instruction datasets from "seed instructions + model continuations", but the model output must pass two gates: **rule filtering** (too short, too long, about images/files, starting with punctuation, starting with non-English, ...) and **similarity dedup** (drop anything whose ROUGE-L similarity to an existing instruction exceeds a threshold). Survivors join the pool, and the next round samples from that pool to build prompts.

This problem turns the pipeline into a pure function: given one raw model response, the seed instructions, and **per-candidate precomputed similarity vectors against the pool**, return the per-segment filtering ledger, the dedup ledger, the final instruction list and the next prompt. Similarity computation itself is out of scope (`eval-rouge-score` covers ROUGE-L scoring); here it is an explicit input used only for the threshold decision.

## Rules

Signature: `solve(response, seed_instructions, sims, finish_reason="stop", threshold=0.7, top_k=10, prompt_instructions=None, classification=False)`.

**1. Truncation and splitting.** `response is None` or `finish_reason == "length"` yields an empty result (the model was cut off, and upstream drops the whole batch). Otherwise split on `\n\d+\s?\. `: newline, digits, **at most one** whitespace, a dot and **exactly one space**. So `\n9. x`, `\n9.  x` (two spaces after the dot), `\n9 . x`, `\n9\t. x` and `\n9\n. x` all split, while `\n9.x` (no space after the dot) and `\n9  . x` (two whitespace characters before the dot) do not. The leading segment that has no such prefix is **kept** and filtered like any other.

**2. Normalization.** Collapse `\s+` to a single space, `strip()`, then `capitalize()`. `capitalize()` lowercases **everything but the first letter**, so `WRITE A PYTHON SCRIPT` becomes `Write a python script` and `Write A Program To ...` becomes `Write a program to ...`.

**3. Seven gates** (in upstream order; report the first hit as the reason):

- `empty`: the normalized string is empty;
- `too_short`: `len(inst.split()) <= 3`;
- `too_long`: `len(inst.split()) > 150`;
- `keyword`: any of `image/images/graph/graphs/picture/pictures/file/files/map/maps/draw/plot/go to` matched as `\b(word)\b`, case-insensitive;
- `write_a_program`: starts with `"Write a program"` (checked **after** `capitalize()`, so `Write A Program` is caught too);
- `punct`: the first character is in `string.punctuation`;
- `non_ascii`: the first character is not ASCII.

Note that `\b` is a regex word boundary: `profile` does not hit `file`, and a `file` immediately followed by a CJK character has no boundary after it either, so it does not hit. `go  to` (two spaces) is collapsed to `go to` by normalization before the keyword scan, so it hits.

**4. Threshold dedup.** `sims[i]` is the similarity vector of the `i`-th first-stage survivor against the current pool, where the pool is the seed instructions plus the **already accepted** machine instructions (rejected ones never join). Therefore the vector length must equal the pool size at that moment, otherwise raise `ValueError`.

- A non-empty vector with `max(vec) > threshold` is dropped (**strictly greater**; equal to the threshold passes);
- On acceptance the instruction is appended to the pool;
- `avg_similarity` is the vector mean; with an empty pool `max_similarity` is `None` and `avg_similarity` is `0.0` (no division).

`sims=None` means no similarities are supplied: the dedup stage is skipped and every first-stage survivor is accepted in order, with `max_similarity=None`, `avg_similarity=0.0` and `most_similar={}`.

**5. Most similar entries.** Only **accepted** candidates get `most_similar`: sort the vector by `(score, index)` ascending (equivalent to a stable ascending sort), take the last `top_k` and reverse them, giving a dict ordered by score descending with larger indices first on ties. Fewer than `top_k` entries means all of them.

**6. Prompt assembly** (when `prompt_instructions` is not `None`). The header is one of two: with `classification=True`, `Come up with a series of classification tasks. Try to specify the possible output labels when possible.`, otherwise `Come up with a series of tasks:`. Then one line per instruction as `{i+1}. {text}`, where `text` gets `\s+` collapsed, `strip()`ed and `rstrip(":")`ed (trailing colons only, which can leave a space behind a colon). The final line is `{n+1}.` with **no trailing newline**.

## Example

```text
input: response = "Here are some tasks:\n1. Write a short poem about the sea\n2. be nice\n"
       seed_instructions = ["seed one"], sims = None
output: segments = [{"raw": "Here are some tasks:", "instruction": "Here are some tasks:",
                     "kept": true, "reason": null},
                    {"raw": " Write a short poem about the sea", ...},
                    {"raw": "be nice\n", "instruction": "Be nice", "kept": false,
                     "reason": "too_short"}]
        candidates  = ["Here are some tasks:", "Write a short poem about the sea"]
        instructions = same as candidates (sims=None accepts everything)
        prompt = null
```

```text
input: prompt_instructions = ["Summarize   this   text", "Rate it:", "Weird: :"]
output: prompt = "Come up with a series of tasks:\n1. Summarize this text\n2. Rate it\n3. Weird: \n4."
```

## Constraints

- `response` is a string or `None`; `finish_reason` is a string; `seed_instructions` is a list of strings (possibly empty); `sims` is a list of real-number lists or `None`, and the reals must not contain `nan` / `inf`.
- `threshold` must be a finite real in `[0, 1]` and `bool` is rejected; `top_k` must be an `int` `>=1` (`bool` rejected); `classification` must be a `bool`.
- `sims` must have one entry per first-stage survivor, and each entry must match the pool size at that moment.
- The returned keys are always `segments` / `candidates` / `dedup` / `instructions` / `prompt`; each `segments` row holds `raw` / `instruction` / `kept` / `reason`, each `dedup` row holds `instruction` / `accepted` / `max_similarity` / `avg_similarity` / `most_similar`.
- Inputs must not be mutated (the seed pool is copied before growing).
