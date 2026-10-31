# Reference Approach: Rendering and Byte Ownership

## Keep three coordinate systems separate

Python string indices count Unicode code points, UTF-8 offsets count bytes, and model token indices form a third coordinate system. This problem outputs only byte coordinates, not model token coordinates. No normalization is performed: `é` and `e\u0301` retain their original encodings.

Content containing literal `<|im_end|>` remains content. Do not infer boundaries afterward using regular expressions, `find`, or `rfind`: repeated content, empty content, and literal markers make that ambiguous.

## Reference implementation

1. Validate flag compatibility, the nonempty continuation requirement, record structure, and roles.
2. Append header, content, and terminator fragments. Encode each fragment once; keep string fragments for a final `join` and bytes for `input_bytes`.
3. Record byte-list lengths immediately before and after content to obtain its half-open span. Label syntax -1 and content 0/1/2. Only role 2 produces `True` mask entries.
4. Continuation skips only the final terminator fragment. A generation header is a new syntax fragment without content or an additional span.

For $B$ output bytes and $M$ messages, time and total space are $O(B+M)$, including the return value. Avoid repeated concatenation of a growing string.

## Independent tests

The oracle first labels individual Unicode characters and records character boundaries. It then expands each character into UTF-8 byte events and converts each boundary using the encoded length of its character prefix. It **does not reuse the reference implementation's message-fragment byte-offset accumulation**.

There are 15 `check_*` functions: basic example, empty conversations and generation, continuation for all three roles, empty content, repeated content, literal markers, Unicode scalars and combining characters, trailing whitespace, syntax masks, invalid roles, invalid records, invalid flag combinations, input invariance, locally seeded randomized character comparisons, and the maximum 20 records with 1000 code points each. Exact keys and Python element types are checked, so integer 0/1 masks cannot substitute for booleans.

Both submit selftest and direct `.venv-eval` execution passed 15/15 before the example dump used in the descriptions was executed. The basic example has spans `[(17, 19), (52, 55)]` and 66 bytes; an empty-conversation generation header has 22 bytes; continuation of the single ideograph `U+7B54` followed by a space and a newline has span `(22, 27)`.

## Upstream boundaries and evidence status

The local Transformers 5.20.2 source was inspected:

- `tokenization_utils_base.py:3043–3055,3122–3128,3176–3199`: generation and continuation are mutually exclusive; rendering precedes tokenization; assistant character intervals require `char_to_token` mapping to model tokens.
- `utils/chat_template_utils.py:534–537,557–563,653–678,697–714`: Jinja rendering, character `len`, and continuation truncation.

These records motivate separating rendering, character coordinates, and model token coordinates. They **do not establish this problem's format as any model's complete default template**. The problem does not execute Jinja or reproduce general upstream template execution, character-to-model-token mapping, or a model-specific supervision policy. Per-byte role IDs, assistant-content-only masks, no implicit system message, and continuation for any final role with unchanged trailing whitespace are explicit problem rules, not claims about every upstream template.
