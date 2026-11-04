# CPT context-aware prompt tuning

Context-aware Prompt Tuning (CPT, arXiv 2410.17222) ships in HF PEFT
(`peft/tuners/cpt/model.py`, `CPTEmbedding`). CPT combines ICL, Prompt Tuning and
adversarial optimization: only **specific context token embeddings** of a few-shot
prompt are fine-tuned (via a learned delta), a **projected gradient descent** keeps
the delta inside per-token epsilon balls to prevent overfitting, and an **exponential
decay weighted loss** mitigates recency bias (later examples dominating the loss).

## Mechanism

### 1. Token type-mask family

`cpt_tokens_type_mask` is a list of integers, interpreted via `m = mask % 4`:

| m | meaning | projection radius |
|---|---|---|
| 1 | input-template (format tokens) | format eps |
| 2 | input (example content tokens) | input eps |
| 3 | output-template (format tokens) | format eps |
| 0 (and mask > 0) | example-label groups (group ids) | frozen (1e-10) |

The trainable mask is `m in {1, 2, 3}` (the backward hook in `set_updated_tokens`
multiplies gradients by exactly this 0/1 mask). Label groups `%4==0` are not updated,
but their **group ids drive the decay-loss weighting**.

### 2. Epsilon radii and projection

Both epsilons are normalized by model width: `eps * sqrt(token_dim / 2048)`.

`get_projection` projects the delta weights row-wise onto an L2 ball:

```
token_norm = ||delta_row||_2
scale = eps_row / clamp(token_norm, min=eps_row)   # only rows with norm > 0
delta_row *= scale
```

i.e. **shrink only, never grow**: rows with norm < eps are untouched; rows with
norm == 0 stay exactly zero (`projection_mask = norm > 0`).

### 3. Forward

```
output = Embedding(frozen prompt ids) + DeltaEmbedding(projected delta)
```

The prompt embedding is initialized from `cpt_token_ids` against the base word
embeddings and frozen; only the delta trains.

### 4. Decay loss

Standard next-token shift of logits/labels, per-token cross entropy
(`ignore_index=-100`), then a weighted mean. Weighting rule
(`opt_weighted_loss_type == "decay"`):

- From the shifted mask, take the group ids (`%4==0` and `>0`), made unique and
  iterated in **reverse order**;
- The first (largest id) group gets weight 1; each later group multiplies
  `opt_loss_decay_factor`;
- Smaller group ids therefore decay geometrically — the paper's counterweight to
  recency bias;
- Invalid (-100) positions participate nowhere: both weights and losses average over
  valid tokens only.

## Judging domain

Single entry `solve(mode, **kw)`, five modes: `build_masks` / `get_epsilon` /
`project` / `embed` / `decay_loss`. Backward hooks, `gather_params_ctx` and vocab
initialization I/O are out of scope; numeric anchors live in the float64 domain, with
the fp32 epsilon computation anchored (official `torch.Tensor([...])` defaults to fp32).
