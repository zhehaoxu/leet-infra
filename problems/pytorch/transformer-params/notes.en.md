# Approach

## What the interviewer is after

"Given a Transformer configuration, tell me how many parameters it has" is a frequent whiteboard question
for infrastructure/training roles. It tests whether you really understand the weight matrix shapes of each
module, rather than memorizing "GPT-2 has 120M". This problem fixes the structure to one clear accounting
and asks you to count the four parameter families — **attention, FFN, LayerNorm, embedding** — separately
and then add them up.

## Item-by-item breakdown

Learnable weights inside one decoder block:

1. **Self-Attention**: the $q,k,v$ projections map $h\to h$, and the output projection $o$ is also $h\to h$;
   that is $4$ matrices of $h\times h$ → $4h^2$. Note this is independent of the number of heads: no matter
   how many heads you split into, `q_proj` always has shape `[hidden, hidden]`.
2. **FFN (MLP)**: up-project $h\to f$ then back down $f\to h$ → $2hf$.
3. **LayerNorm ×2**: each LN has a weight and a bias with $h$ entries each → $2\times 2h = 4h$.

Adding the three gives $P_{\text{layer}} = 4h^2 + 2hf + 4h$. Multiply by the number of layers to get the
parameters of all blocks.

## Embedding and output head

- The input token embedding is `vocab × hidden`.
- The output LM head is also `vocab × hidden`. **Tied embedding** (weight sharing) makes both use the same
  table, so it is counted once; when untied, add another `vocab·h`. GPT-2 ties them.
- The network ends with one final LayerNorm: $2h$.

## Order-of-magnitude intuition

For GPT-2 small ($h=768, f=3072, L=12$): the block parameters $84.97$M dwarf the embedding $38.6$M, and
$4h^2$ and $2hf$ each account for a large share of the block parameters — so with $f=4h$, attention's
$4h^2$ and FFN's $2h\cdot 4h = 8h^2$ are in roughly a $1:2$ ratio. This is the empirical origin of "the FFN
is the parameter hog".

## Common mistakes

- Forgetting the LayerNorm / final LN: a small share on small models, but cross-checking will catch it.
- Getting tied/untied backwards: untied always adds exactly one `vocab×h` table over tied.
- Multiplying parameters by the number of heads: MHA's projections are always $4h^2$, independent of heads.

## Judging methodology

`test_cases.py` cross-checks the formula implementation against an independent reference `_ref` that
**accumulates matrix by matrix**, ensuring both derivations agree; `check_untied_adds_head` separately
verifies that the tied/untied difference is exactly one embedding table.
