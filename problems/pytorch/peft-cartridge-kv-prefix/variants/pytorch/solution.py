"""PEFT: Cartridge KV-prefix adapter (official PEFT tuners/cartridge).

Six entry points mirroring PEFT PR #2953 (arXiv:2506.06266):

- ``prompt_embeddings_from_past_key_values``: pack a legacy KV cache
  (list of per-layer ``(key, value)`` pairs, each ``[1, H, S, D]``) into the
  flattened prefix-embedding layout ``[NVT, L*2*H*D]``.
- ``CartridgeEncoder``: parameterized prefix KV cache with a frozen
  (attention-sink) head segment and a trainable tail segment.
- ``compose_cartridge_adapters``: concatenate adapters' prompt embeddings.
"""

import torch


def prompt_embeddings_from_past_key_values(past_key_values, *, num_virtual_tokens):
    """Convert a legacy KV cache into flattened prefix prompt embeddings.

    Official layout (utils.py): ``packed[:, 2*layer] = k[0, :, :nvt, :].T`` and
    ``packed[:, 2*layer+1] = v[0, :, :nvt, :].T``, then ``reshape(nvt, -1)``
    giving ``[num_virtual_tokens, num_layers * 2 * num_heads * head_dim]``.
    """
    legacy = past_key_values
    if len(legacy) == 0:
        raise ValueError("Empty `past_key_values`.")

    num_layers = len(legacy)
    key0, _ = legacy[0]
    if key0.ndim != 4:
        raise ValueError(f"Expected key/value tensors with rank 4, got key.ndim={key0.ndim}.")
    if key0.shape[0] != 1:
        raise ValueError(
            "This helper expects `past_key_values` from a single-sequence prefill (batch=1). "
            f"Got batch={key0.shape[0]}."
        )
    num_heads = key0.shape[1]
    seq_len = key0.shape[2]
    head_dim = key0.shape[3]
    if seq_len < num_virtual_tokens:
        raise ValueError(f"Need at least {num_virtual_tokens} cached tokens, got {seq_len}.")

    packed = torch.empty(
        num_virtual_tokens,
        num_layers * 2,
        num_heads,
        head_dim,
        device=key0.device,
        dtype=key0.dtype,
    )
    for layer_idx, (k, v) in enumerate(legacy):
        if k.shape[:2] != (1, num_heads) or v.shape[:2] != (1, num_heads):
            raise ValueError("Inconsistent head shapes across layers in `past_key_values`.")
        if k.shape[2] < num_virtual_tokens or v.shape[2] < num_virtual_tokens:
            raise ValueError("Not enough cached tokens in `past_key_values` for the requested cartridge length.")
        if k.shape[3] != head_dim or v.shape[3] != head_dim:
            raise ValueError("Inconsistent head_dim across layers in `past_key_values`.")
        packed[:, 2 * layer_idx] = k[0, :, :num_virtual_tokens, :].transpose(0, 1).contiguous()
        packed[:, 2 * layer_idx + 1] = v[0, :, :num_virtual_tokens, :].transpose(0, 1).contiguous()

    return packed.reshape(num_virtual_tokens, -1)


class CartridgeEncoder(torch.nn.Module):
    """A parameterized prefix KV cache (official model.py ``CartridgeEncoder``).

    Stores the flattened KV prefix ``[num_virtual_tokens, num_layers*2*token_dim]``
    directly as parameters (no MLP reparameterization, unlike prefix tuning).
    The first ``num_frozen_tokens`` rows are a non-trainable parameter (the
    paper recommends freezing the first token as an attention sink).
    """

    def __init__(self, num_virtual_tokens, num_layers, token_dim, num_frozen_tokens=1, dtype=torch.float32):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.num_layers = num_layers
        self.token_dim = token_dim
        hidden = num_layers * 2 * token_dim
        self.hidden = hidden
        num_frozen_tokens = int(num_frozen_tokens)
        if num_frozen_tokens < 0 or num_frozen_tokens > num_virtual_tokens:
            raise ValueError(
                f"`num_frozen_tokens` must be in [0, num_virtual_tokens], got {num_frozen_tokens} for "
                f"num_virtual_tokens={num_virtual_tokens}."
            )

        self.num_frozen_tokens = num_frozen_tokens
        self.num_trainable_tokens = num_virtual_tokens - num_frozen_tokens

        if self.num_frozen_tokens:
            frozen = torch.empty(self.num_frozen_tokens, hidden, dtype=dtype)
            self.frozen_embedding = torch.nn.Parameter(frozen, requires_grad=False)
        else:
            self.frozen_embedding = None

        trainable = torch.empty(self.num_trainable_tokens, hidden, dtype=dtype)
        self.trainable_embedding = torch.nn.Parameter(trainable, requires_grad=True)

        self.reset_parameters()

    @property
    def weight(self):
        if self.frozen_embedding is None:
            return self.trainable_embedding
        return torch.cat([self.frozen_embedding, self.trainable_embedding], dim=0)

    def reset_parameters(self):
        # Match `torch.nn.Embedding` initialization (normal with std=1).
        with torch.no_grad():
            if self.frozen_embedding is not None:
                torch.nn.init.normal_(self.frozen_embedding)
            torch.nn.init.normal_(self.trainable_embedding)

    def load_prompt_embeddings(self, prompt_embeddings):
        """Split a flat prompt-embeddings tensor into frozen/trainable segments."""
        if prompt_embeddings.ndim != 2 or prompt_embeddings.shape[0] != (
            self.num_frozen_tokens + self.num_trainable_tokens
        ):
            raise ValueError(
                "Invalid `prompt_embeddings` shape. Expected "
                f"({self.num_frozen_tokens + self.num_trainable_tokens}, hidden), got {tuple(prompt_embeddings.shape)}."
            )
        with torch.no_grad():
            if self.frozen_embedding is not None:
                self.frozen_embedding.copy_(prompt_embeddings[: self.num_frozen_tokens].to(self.frozen_embedding.device))
                trainable_part = prompt_embeddings[self.num_frozen_tokens :]
            else:
                trainable_part = prompt_embeddings
            self.trainable_embedding.copy_(trainable_part.to(self.trainable_embedding.device))

    def forward(self, prefix_tokens):
        # Ignore token ids; they exist for prompt-learning uniformity.
        batch_size = prefix_tokens.shape[0]
        return self.weight.unsqueeze(0).expand(batch_size, -1, -1)


COMPAT_ATTRS = ("task_type", "token_dim", "num_layers", "num_attention_heads", "num_transformer_submodules")


def compose_cartridge_adapters(adapters):
    """Compose CARTRIDGE adapters by concatenating prompt embeddings (official utils.py).

    ``adapters`` is a list of dicts with keys ``prompt_embeddings`` (tensor) and
    ``config`` (dict, must contain ``peft_type`` and the compat attributes plus
    ``num_frozen_tokens``). Returns ``(composed_embeddings, num_frozen_tokens)``:
    embeddings are concatenated along dim 0 and the frozen-token count is taken
    from the first adapter only (matches a single attention-sink prefix).
    """
    if len(adapters) < 2:
        raise ValueError("Need at least 2 adapters to compose.")

    configs = [a["config"] for a in adapters]
    for cfg in configs:
        if cfg.get("peft_type") != "CARTRIDGE":
            raise ValueError(f"Adapter is not a CARTRIDGE adapter (got {cfg.get('peft_type')}).")

    base = configs[0]
    for attr in COMPAT_ATTRS:
        for cfg in configs[1:]:
            if cfg.get(attr) != base.get(attr):
                raise ValueError(f"Incompatible CARTRIDGE configs for attribute '{attr}'.")

    prompt_embeddings = [a["prompt_embeddings"] for a in adapters]
    composed = torch.cat(prompt_embeddings, dim=0)

    num_frozen_tokens = base.get("num_frozen_tokens")
    return composed, num_frozen_tokens


def solve(mode=None, **kw):
    if mode == "pack":
        return prompt_embeddings_from_past_key_values(
            kw["past_key_values"], num_virtual_tokens=kw["num_virtual_tokens"]
        )
    if mode == "encoder_new":
        torch.manual_seed(kw["seed"])
        return CartridgeEncoder(
            kw["num_virtual_tokens"], kw["num_layers"], kw["token_dim"],
            num_frozen_tokens=kw.get("num_frozen_tokens", 1),
            dtype=kw.get("dtype", torch.float32),
        )
    if mode == "encoder_weight":
        return kw["encoder"].weight
    if mode == "encoder_params":
        enc = kw["encoder"]
        return enc.frozen_embedding, enc.trainable_embedding
    if mode == "encoder_load":
        return kw["encoder"].load_prompt_embeddings(kw["prompt_embeddings"])
    if mode == "encoder_forward":
        return kw["encoder"](kw["prefix_tokens"])
    if mode == "compose":
        return compose_cartridge_adapters(kw["adapters"])
    raise ValueError(f"unknown mode {mode!r}")
