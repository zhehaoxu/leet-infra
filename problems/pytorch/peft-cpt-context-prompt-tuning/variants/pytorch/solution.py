"""Context-aware Prompt Tuning (CPT) — HuggingFace PEFT cpt tuner.

Reproduces the pure-function core of peft/src/peft/tuners/cpt/model.py (CPTEmbedding,
arXiv 2410.17222). CPT refines *context* token embeddings of an ICL prompt with
projected gradient descent: only tokens whose type-mask entry participates get a
learned delta, deltas are projected onto per-token epsilon balls, and the training
loss weights example tokens with an exponential decay by group id.

Token type-mask semantics (official): mask values are integers; with m = mask % 4,
- m == 1: input-template tokens (format eps)
- m == 2: input/example tokens (input eps)
- m == 3: output-template tokens (format eps)
- m == 0 (and mask > 0): example-label groups — excluded from updates (eps ~ 0) and
  used as decay-loss group ids.
"""

import torch
import torch.nn.functional as F

MIN_VALUE = 1e-10


def solve(mode, **kw):
    if mode == "build_masks":
        # Official set_updated_tokens: trainable iff mask % 4 in {1, 2, 3}.
        cpt_tokens_type_mask = kw["cpt_tokens_type_mask"]
        t = torch.tensor(cpt_tokens_type_mask, dtype=torch.long)
        mask_input_template = torch.remainder(t, 4) == 1
        mask_input = torch.remainder(t, 4) == 2
        mask_output_template = torch.remainder(t, 4) == 3
        mask = mask_input_template | mask_input | mask_output_template
        return {"mask": mask.view(-1, 1)}

    if mode == "get_epsilon":
        # Official get_epsilon: per-token projection radii. Format/template tokens
        # (m == 1 or 3) get the (smaller) format epsilon; input tokens (m == 2) get
        # the input epsilon; everything else (label groups, m == 0) is pinned at
        # MIN_VALUE (effectively frozen). Both epsilons scale with sqrt(dim / 2048).
        cpt_tokens_type_mask = kw["cpt_tokens_type_mask"]
        token_dim = kw["token_dim"]
        input_eps, format_eps = kw["opt_projection_epsilon"], kw["opt_projection_format_epsilon"]
        normalized_format_eps = format_eps * torch.sqrt(torch.tensor([token_dim / 2048], dtype=torch.float32))
        normalized_input_eps = input_eps * torch.sqrt(torch.tensor([token_dim / 2048], dtype=torch.float32))
        m = torch.tensor(cpt_tokens_type_mask, dtype=torch.long)
        epsilon = torch.ones(len(cpt_tokens_type_mask), dtype=torch.float32) * MIN_VALUE
        epsilon[(m > 0) & (torch.remainder(m, 4) == 1)] = normalized_format_eps
        epsilon[(m > 0) & (torch.remainder(m, 4) == 3)] = normalized_format_eps
        epsilon[(m > 0) & (torch.remainder(m, 4) == 2)] = normalized_input_eps
        return epsilon

    if mode == "project":
        # Official get_projection: per-token L2 norm ball. Rows with norm > 0 are
        # rescaled by epsilon / clamp(norm, min=epsilon) — i.e. shrink to the ball,
        # never grow. Rows with norm == 0 stay exactly zero.
        delta = kw["delta"]
        epsilon = kw["epsilon"]
        new_w = delta.clone()
        token_norm = torch.norm(new_w, p=2, dim=1)
        projection_mask = token_norm > 0
        if torch.any(projection_mask):
            new_w[projection_mask] *= (
                epsilon[projection_mask] / (token_norm[projection_mask].clamp(min=epsilon[projection_mask]))
            ).view(-1, 1)
        return new_w

    if mode == "embed":
        # Official forward: prompt embedding (frozen) + delta embedding.
        indices = kw["indices"]
        embedding_w = kw["embedding_w"]
        delta_w = kw["delta_w"]
        return F.embedding(indices, embedding_w) + F.embedding(indices, delta_w)

    if mode == "decay_loss":
        # Official calculate_loss: shifted token-level cross entropy (ignore_index
        # -100), then weighted mean. Weight for each valid token: 1 by default; with
        # "decay" weighting, example-label groups (mask % 4 == 0) get exponential
        # weights by group id: iterating the group ids in REVERSE order, the first
        # (largest id) group has weight 1 and each later group multiplies the decay
        # factor — smaller group ids get geometrically smaller weights.
        logits, labels = kw["logits"], kw["labels"]
        cpt_type_mask = kw["cpt_type_mask"]
        decay_factor = kw["opt_loss_decay_factor"]
        weighted_loss_type = kw.get("opt_weighted_loss_type", "none")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_cpt_type_mask = cpt_type_mask[..., 1:].contiguous()

        shift_labels_bool = (shift_labels.clone().detach() != -100).bool()
        batch_size, seq_length, vocab_size = shift_logits.shape

        loss = F.cross_entropy(
            shift_logits.view(batch_size * seq_length, vocab_size),
            shift_labels.view(batch_size * seq_length),
            reduction="none", ignore_index=-100,
        )
        loss = loss.view(batch_size, seq_length)

        shift_labels_weights = shift_labels_bool.clone().detach().float()
        for i in range(batch_size):
            idx_labels = (shift_cpt_type_mask[i] > 0) & (shift_cpt_type_mask[i] % 4 == 0)
            labels_ids = shift_cpt_type_mask[i][idx_labels].unique()
            exponential_decay = torch.ones_like(shift_cpt_type_mask[i]).to(device=logits.device).float()
            decay_value = 1.0
            for label_mask_idx in torch.flip(labels_ids, [0]):
                exponential_decay[shift_cpt_type_mask[i] == label_mask_idx] = decay_value
                decay_value *= decay_factor
            if weighted_loss_type == "decay":
                shift_labels_weights[i] *= exponential_decay

        loss = (loss[shift_labels_bool] * shift_labels_weights[shift_labels_bool]).mean()
        return loss

    raise ValueError(f"unknown mode: {mode}")


CHECKS = [
    "check_build_masks",
    "check_get_epsilon",
    "check_project",
    "check_embed",
    "check_decay_loss",
]


def check(solve):
    for name in CHECKS:
        globals()[name](solve)
