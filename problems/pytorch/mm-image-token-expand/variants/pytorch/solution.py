import torch


def solve(text_embeds, labels, attention_mask, image_mask, image_features):
    if torch.any(image_mask & ~attention_mask) or torch.any(image_mask.sum(dim=1) != 1):
        raise ValueError("Each row must have exactly one image marker, at a valid position")

    embed_rows, label_rows, type_rows = [], [], []
    for b, features in enumerate(image_features):
        active = attention_mask[b]
        text = text_embeds[b, active]
        row_labels = labels[b, active]
        index = torch.nonzero(image_mask[b, active], as_tuple=True)[0].item()
        count = features.shape[0]
        embed_rows.append(torch.cat((text[:index], features, text[index + 1:]), dim=0))
        label_rows.append(torch.cat((
            row_labels[:index], labels.new_full((count,), -100), row_labels[index + 1:]
        )))
        types = labels.new_zeros(text.shape[0] - 1 + count)
        types[index:index + count] = 1
        type_rows.append(types)

    B, _, D = text_embeds.shape
    length = max(row.shape[0] for row in embed_rows)
    outputs = {
        "inputs_embeds": text_embeds.new_zeros((B, length, D)),
        "labels": labels.new_full((B, length), -100),
        "attention_mask": attention_mask.new_zeros((B, length)),
        "position_ids": labels.new_zeros((B, length)),
        "mm_token_type_ids": labels.new_zeros((B, length)),
    }
    for b, row in enumerate(embed_rows):
        size = row.shape[0]
        outputs["inputs_embeds"][b, :size] = row
        outputs["labels"][b, :size] = label_rows[b]
        outputs["attention_mask"][b, :size] = True
        outputs["position_ids"][b, :size] = torch.arange(size, device=labels.device)
        outputs["mm_token_type_ids"][b, :size] = type_rows[b]
    return outputs
