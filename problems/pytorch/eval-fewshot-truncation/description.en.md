# Evaluation: Few-shot Prompt Truncation under a Token Budget

## Problem

When an evaluation harness (LM-eval-harness, OpenCompass and friends) builds a few-shot prompt it must squeeze the demonstrations into the model's context budget: demonstrations are appended **in a fixed order**, and the moment "adding one more would exceed the budget" holds, the loop **stops** — note, stops, not "skip this one and try the shorter next one". At the same time the query template and the **label tail** (for instance `"Answer:"`, after which the model must continue) can never be truncated, or the output format collapses. Wang et al., *How does in-context learning work? A framework for understanding the successes and failures of LLMs* (arXiv:2201.11903) formalised this demonstration-selection-plus-budget-truncation procedure.

To remove the dependency on a HuggingFace tokenizer (the judge environment is offline), the problem ships its own **toy tokenizer** (see the code template): split on whitespace; a word present in `VOCAB` is worth one token, a word that is not gets expanded into **per-character** tokens (one `<unk>` per character).

## Rules

- Inputs: `demos` (`list[str]`, demonstration texts in fixed order, possibly empty), `query` (`str`, the template), `tail` (`str`, the label tail), `budget` (`int`, token ceiling for the whole prompt).
- Assembly order: the kept demonstrations (in the given order) → `query` → `tail`, joined by single spaces. Because the tokenizer only understands whitespace, **joining adds no tokens** and the total is the sum of the parts' token counts.
- The budget is occupied starting from `query` and `tail`: \(u_0 = \text{tok}(\text{query}) + \text{tok}(\text{tail})\); they are **never truncated** — even if \(u_0 > budget\) they stay in full and the over-budget length is reported honestly.
- Greedy rule: for \(i = 0, 1, 2, \dots\)
  \[ u_i + \text{tok}(\text{demo}_i) \le budget \;\Rightarrow\; \text{keep demo } i,\ u_{i+1} = u_i + \text{tok}(\text{demo}_i) \]
  otherwise **stop immediately** and drop every remaining demonstration, even shorter ones that would still fit. Equality with the budget counts as "keep".
- Return the pair `(kept, final_len)`: `kept` is an int64 tensor of shape \((m,)\) holding the kept demonstration indices (ascending, and necessarily a prefix starting at 0); `final_len` is a Python int equal to the token count of the assembled prompt.
- The toy tokenizer and `VOCAB` are given in the code template and must be used as defined; no real tokenizer may be introduced.

## Example

With `budget = 15`, `query = "Question:"` (1 token) and `tail = "Answer:"` (1 token):

```
demo0 = "the cat sat on the mat"   -> 6 tokens (all in VOCAB)
demo1 = "the dog is very nice"     -> 8 tokens ("very" is not in VOCAB -> v,e,r,y)
demo2 = "yes"                      -> 1 token
demo3 = "no"                       -> 1 token

u0 = 1 + 1 = 2
add demo0: 2 + 6 = 8  <= 15  -> keep
add demo1: 8 + 8 = 16 >  15  -> stop (demo2/demo3 are 1 token each, still not picked up)

kept = [0]        final_len = 8
assembled prompt = "the cat sat on the mat Question: Answer:" (exactly 8 tokens)
```

## Constraints

- CPU; pure integer arithmetic, no randomness.
- \(0 \le budget\); `demos` may be an empty list and a single demo may consist only of whitespace (0 tokens).
- **Degenerate properties**: empty `demos` ⇒ `kept = []` and `final_len = tok(query) + tok(tail)`; a budget at least as large as everything ⇒ all kept; `budget < tok(query) + tok(tail)` ⇒ no demo is kept and `final_len` truthfully exceeds `budget`; a budget exactly equal to some running total ⇒ that demo is kept.
- `kept` must be a prefix `[0, 1, ..., m-1]`; indices that skip a middle demonstration are wrong by construction.
