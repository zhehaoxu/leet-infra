"""Cartridge (KV-cache-parameterized prefix adapter, arXiv:2506.06266, PEFT tuners/cartridge).

Unlike prefix tuning (which learns an MLP that generates the KV prefix), Cartridge
stores the flattened prefix KV cache directly as trainable parameters, with the first
`num_frozen_tokens` rows frozen (attention sink).

Accounts (official PEFT implementation, PR #2953):
- pack a legacy KV cache: per layer (k, v), each [1, H, S, D] -> packed
  [NVT, L*2, H, D] where column 2*l holds k[0,:,:NVT,:].T and 2*l+1 holds v[...].T,
  then reshape to [NVT, L*2*H*D]; validate empty cache / rank / batch=1 /
  seq_len >= NVT / per-layer head & head_dim consistency / per-layer cache length
- CartridgeEncoder: [NVT, L*2*token_dim] parameters; first num_frozen_tokens rows
  in a requires_grad=False parameter (Embedding-style normal std=1 init, frozen
  drawn BEFORE trainable from the same RNG stream); weight = cat([frozen, trainable])
- load_prompt_embeddings: split rows [0:nft] -> frozen, [nft:] -> trainable
- forward: ignore token ids, weight.unsqueeze(0).expand(batch, -1, -1)
- compose adapters: cat(prompt_embeddings, dim=0), frozen count from the FIRST
  adapter only, config compat on task_type/token_dim/num_layers/num_attention_heads/
  num_transformer_submodules, peft_type must be CARTRIDGE
"""
import torch


def prompt_embeddings_from_past_key_values(past_key_values, *, num_virtual_tokens):
    raise NotImplementedError


class CartridgeEncoder(torch.nn.Module):
    def __init__(self, num_virtual_tokens, num_layers, token_dim, num_frozen_tokens=1, dtype=torch.float32):
        raise NotImplementedError

    @property
    def weight(self):
        raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError

    def load_prompt_embeddings(self, prompt_embeddings):
        raise NotImplementedError

    def forward(self, prefix_tokens):
        raise NotImplementedError


COMPAT_ATTRS = ("task_type", "token_dim", "num_layers", "num_attention_heads", "num_transformer_submodules")


def compose_cartridge_adapters(adapters):
    raise NotImplementedError


def solve(mode=None, **kw):
    raise NotImplementedError
