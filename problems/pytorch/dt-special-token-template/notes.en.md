# Notes: Assemble Templates by Positional Origin

## Reference approach

Validate `style` first, then determine pair existence with `ids_b is not None`. Do not use `if ids_b`: an empty B must retain its pair separators.

Represent the output as a few segments carrying `(IDs, type_id, special_flag, sequence_id)`. Original A has attributes `(0,0,0)`; original B has `(1,0,1)` for BERT and `(0,0,1)` for RoBERTa. Inserted segments have special flag 1 and sequence ID `None`. Extend all four output lists in template order.

Disabling insertion simply removes inserted segments; it does not change original type or sequence ownership. An input ID equal to CLS/SEP remains an ordinary original position, so special identity must not be inferred through numeric membership. Extend fresh lists rather than appending to or modifying the input lists.

## Correctness

Every original A/B position enters its segment exactly once and in order. Each template insertion also appears exactly once, establishing the required ID sequence.

All four attributes of a segment are expanded by the same length, so each output index describes the same origin event across all fields. Original positions and insertion events remain distinct even when IDs collide. Pair existence is checked independently of B's length, so empty B still triggers pair insertion. The same argument covers empty A, both sides empty and disabled insertion.

Let $n_A,n_B$ be the input lengths, with absent B counted as zero, and let $s$ count insertions: 0 when disabled, 2 for a single sequence, 3 for a BERT pair and 4 for a RoBERTa pair. Time and output space are $O(n_A+n_B+s)$; excluding the output, there are only a constant number of segment descriptors.

## Independent tests

The reference concatenates segments; the oracle emits one position at a time through states for start, A, A's ending, bridge, B and B's ending. It neither calls the reference nor shares its template assembly helper. Tests compare the complete dictionary, exact keys, list/element types and input immutability.

`--selftest dt-special-token-template --mode submit` passed **12/12**, covering:

- Both nonempty pair templates and the default insertion argument;
- Empty A, absent B, empty B and both sides empty;
- Disabled insertion, ID collisions and equal CLS/SEP IDs;
- Asymmetric lengths, both sides at length 30 and boundary ID values;
- A small exhaustive grid and 150 random inputs with seed 458;
- Invalid style validation even for empty input or disabled insertion;
- Repeated calls and shared A/B input lists without mutation.

## Native fixture provenance and execution

Code and independent tests first passed submit selftests using `.venv-backend/bin/python`. Only afterward were statement examples and native comparisons executed with `.venv-eval/bin/python`. Delivered starter, solution and tests do not depend on tokenizers, transformers or the fixture-generation environment.

Local versions were transformers **5.20.2** and tokenizers **0.23.3**. Sources inspected:

- `transformers/models/bert/tokenization_bert.py:128–135`: `TemplateProcessing` single/pair templates and type assignments.
- `transformers/models/roberta/tokenization_roberta.py:170–175`: `RobertaProcessing` construction.
- `tokenizers/__init__.pyi:279–402,931–967`: Encoding ownership fields and the optional pair in `encode`.

Native checks used a `WordLevel` vocabulary `{'a':11,'c':12,'b':21,'[UNK]':99}` with `WhitespaceSplit`, isolating the template from text tokenization. BERT used the source's `TemplateProcessing` with CLS/SEP 101/102. RoBERTa used `RobertaProcessing(('</s>',2),('<s>',0), trim_offsets=False, add_prefix_space=False)`.

A in `''/'a'/'a c'`, B in `None/''/'b'`, both styles and both insertion flags produced **36** native comparisons, all matching all four reference fields. Literal fixtures include ordinary pairs and `backend.encode('a','')`:

| Field | BERT empty B | RoBERTa empty B |
| --- | --- | --- |
| input_ids | `[101,11,102,102]` | `[0,11,2,2,2]` |
| token_type_ids | `[0,0,0,1]` | `[0,0,0,0,0]` |
| special_tokens_mask | `[1,0,1,1]` | `[1,0,1,1,1]` |
| sequence_ids | `[None,0,None,None]` | `[None,0,None,None,None]` |

All four statement examples were executed before their outputs were recorded, including disabled insertion with ID collisions.

## API boundary: no universal-equivalence claim

The local single-item wrapper in `transformers/tokenization_utils_tokenizers.py:1047` decides whether to pass a pair using `text_pair` truthiness. With a small vocabulary assigning CLS=2, SEP=3 and a=5, an actual `BertTokenizer` call `tokenizer('a','')` returned `[2,5,3]`, whereas `tokenizer.backend_tokenizer.encode('a','')` returned `[2,5,3,3]`.

This exercise explicitly chooses the latter convention: an explicitly empty B remains a pair. Not all HF APIs agree. The exercise models template fields only, not tokenization, added-token registration, normalization, truncation, padding or batch-wrapper behavior.
