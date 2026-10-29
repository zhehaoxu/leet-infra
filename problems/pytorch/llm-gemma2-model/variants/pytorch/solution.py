import math

import torch


_ALLOWED_CONFIG_KEYS = {
    "hidden_size",
    "vocab_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "num_hidden_layers",
    "layer_types",
    "sliding_window",
    "query_pre_attn_scalar",
    "attn_logit_softcapping",
    "final_logit_softcapping",
    "rms_norm_eps",
    "rope_theta",
    "tie_word_embeddings",
}

_LAYER_TYPES = ("sliding_attention", "full_attention")

_LAYER_KEYS = (
    "input_ln_weight",
    "post_attention_ln_weight",
    "pre_feedforward_ln_weight",
    "post_feedforward_ln_weight",
    "q_weight",
    "k_weight",
    "v_weight",
    "o_weight",
    "gate_weight",
    "up_weight",
    "down_weight",
)

_WEIGHT_KEYS = tuple(k for k in _LAYER_KEYS if k.endswith("_weight") and "ln" not in k)
_NORM_KEYS = tuple(k for k in _LAYER_KEYS if "ln" in k)


def _int_option(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _positive_float(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _optional_positive_float(name, value):
    if value is None:
        return None
    return _positive_float(name, value)


def _float_tensor(name, value, ndim):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must be float32 or float64")
    if value.dim() != ndim:
        raise ValueError(f"{name} must be {ndim}-d")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value.to(torch.float64)


def _expect_shape(name, tensor, shape):
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {tuple(shape)}")


def _rms(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * (1.0 + weight)


def solve(
    input_ids: torch.Tensor,
    embed_weight: torch.Tensor,
    final_norm_weight: torch.Tensor,
    layers: list,
    config: dict,
    lm_head_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gemma2 整模型装配：config 推导维度 → 逐层交替掩码 → 顶部 norm → 词表头 → logit softcap。

    input_ids        : (B, S) 整数 token id，取值 [0, vocab)
    embed_weight     : (vocab, hidden) 嵌入表；前向时整体乘 sqrt(hidden)
    final_norm_weight: (hidden,) 栈顶 RMSNorm 的原始 gain w（按 (1 + w) 使用）
    layers           : list[dict]，每层键：
        input_ln_weight, post_attention_ln_weight,
        pre_feedforward_ln_weight, post_feedforward_ln_weight,
        q_weight (Hq*hd, hidden), k_weight/v_weight (Hkv*hd, hidden),
        o_weight (hidden, Hq*hd), gate_weight/up_weight (I, hidden), down_weight (hidden, I)
        （投影均无 bias；四个 gain 同样按 (1 + w) 使用）
    config           : dict，键全部可选（`num_attention_heads` 必填），缺省值按 HF Gemma2Config：
        num_attention_heads  必填，Hq
        num_key_value_heads  缺省 = Hq
        head_dim             缺省 = hidden_size // Hq
        hidden_size          缺省 = embed_weight.shape[1]
        vocab_size           缺省 = embed_weight.shape[0]
        num_hidden_layers    缺省 = len(layers)，且必须相等
        layer_types          缺省第 0 层 sliding、第 1 层 full 交替
        sliding_window       缺省 4096（含当前 token）
        query_pre_attn_scalar 缺省 256，缩放取它的 ** -0.5
        attn_logit_softcapping / final_logit_softcapping 缺省 50 / 30，None 表示关掉
        rms_norm_eps         缺省 1e-6
        rope_theta           缺省 10000.0
        tie_word_embeddings  缺省 True
    lm_head_weight   : (vocab, hidden)；tie 为 True 时必须为 None（复用 embed_weight），
                       tie 为 False 时必填

    数据流（全 float64 内部计算，返回 float32 logits (B, S, vocab)）：
      1. X = embed[input_ids] * sqrt(hidden)；位置 0..S-1 现场生成 cos/sin（重复半维）
      2. 逐层：h = RMSNorm(X)；q/k/v 投影 → 拆头 → RoPE → GQA 复用；
         logits = q·kᵀ · scalar**-0.5 → tanh softcap → 掩码（causal，sliding 再收窄）→ softmax；
         拼回后 o_proj，再 RMSNorm（post_attention gain）→ 加残差；
         第二个 RMSNorm（pre_ff gain）→ GeGLU → down → 再 RMSNorm（post_ff gain）→ 加残差
      3. 顶层 RMSNorm → logits = X @ 词表头ᵀ（tie 时词表头就是 embed_weight）
      4. final_logit_softcapping 非 None 时 logits = cap * tanh(logits / cap)
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a dict")
    unknown = sorted(set(config) - _ALLOWED_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unsupported config key(s): {unknown}")

    embed = _float_tensor("embed_weight", embed_weight, 2)
    vocab, hidden = embed.shape

    if _int_option("vocab_size", config.get("vocab_size", vocab), 1) != vocab:
        raise ValueError("vocab_size does not match embed_weight")
    if _int_option("hidden_size", config.get("hidden_size", hidden), 1) != hidden:
        raise ValueError("hidden_size does not match embed_weight")

    if "num_attention_heads" not in config:
        raise ValueError("config must define num_attention_heads")
    n_heads = _int_option("num_attention_heads", config["num_attention_heads"], 1)

    if "head_dim" in config:
        head_dim = _int_option("head_dim", config["head_dim"], 1)
    else:
        if hidden % n_heads != 0:
            raise ValueError("hidden_size is not divisible by num_attention_heads; set head_dim explicitly")
        head_dim = hidden // n_heads
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even")

    n_kv = _int_option("num_key_value_heads", config.get("num_key_value_heads", n_heads), 1)
    if n_heads % n_kv != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    if not isinstance(layers, (list, tuple)) or len(layers) == 0:
        raise ValueError("layers must be a non-empty list")
    n_layers = len(layers)
    if _int_option("num_hidden_layers", config.get("num_hidden_layers", n_layers), 1) != n_layers:
        raise ValueError("num_hidden_layers does not match len(layers)")

    layer_types = config.get("layer_types")
    if layer_types is None:
        layer_types = ["sliding_attention" if (i + 1) % 2 else "full_attention" for i in range(n_layers)]
    else:
        if not isinstance(layer_types, (list, tuple)) or len(layer_types) != n_layers:
            raise ValueError("layer_types must be a list with one entry per layer")
        for layer_type in layer_types:
            if layer_type not in _LAYER_TYPES:
                raise ValueError(f"layer_types entries must be one of {_LAYER_TYPES}")

    window = _int_option("sliding_window", config.get("sliding_window", 4096), 1)
    pre_attn_scalar = _positive_float("query_pre_attn_scalar", config.get("query_pre_attn_scalar", 256.0))
    attn_softcap = _optional_positive_float("attn_logit_softcapping", config.get("attn_logit_softcapping", 50.0))
    final_softcap = _optional_positive_float("final_logit_softcapping", config.get("final_logit_softcapping", 30.0))
    eps = _positive_float("rms_norm_eps", config.get("rms_norm_eps", 1e-6))
    theta = _positive_float("rope_theta", config.get("rope_theta", 10000.0))
    tied = config.get("tie_word_embeddings", True)
    if not isinstance(tied, bool):
        raise ValueError("tie_word_embeddings must be a bool")

    final_norm = _float_tensor("final_norm_weight", final_norm_weight, 1)
    _expect_shape("final_norm_weight", final_norm, (hidden,))

    if tied:
        if lm_head_weight is not None:
            raise ValueError("lm_head_weight must be None when tie_word_embeddings is True")
        head_weight = embed
    else:
        if lm_head_weight is None:
            raise ValueError("lm_head_weight is required when tie_word_embeddings is False")
        head_weight = _float_tensor("lm_head_weight", lm_head_weight, 2)
        _expect_shape("lm_head_weight", head_weight, (vocab, hidden))

    if not isinstance(input_ids, torch.Tensor):
        raise ValueError("input_ids must be a torch.Tensor")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must be int32 or int64")
    if input_ids.dim() != 2:
        raise ValueError("input_ids must be 2-d (B, S)")
    batch, seq = input_ids.shape
    if batch == 0 or seq == 0:
        raise ValueError("input_ids must be non-empty")
    if int(input_ids.min()) < 0 or int(input_ids.max()) >= vocab:
        raise ValueError("input_ids out of vocabulary range")

    layer_weights = []
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise ValueError(f"layers[{index}] must be a dict")
        missing = [key for key in _LAYER_KEYS if key not in layer]
        if missing:
            raise ValueError(f"layers[{index}] is missing {missing}")
        extra = sorted(set(layer) - set(_LAYER_KEYS))
        if extra:
            raise ValueError(f"layers[{index}] has unsupported key(s) {extra}")
        weights = {key: _float_tensor(f"layers[{index}].{key}", layer[key], 2) for key in _WEIGHT_KEYS}
        for key in _NORM_KEYS:
            weights[key] = _float_tensor(f"layers[{index}].{key}", layer[key], 1)
            _expect_shape(f"layers[{index}].{key}", weights[key], (hidden,))
        intermediate = weights["gate_weight"].shape[0]
        _expect_shape(f"layers[{index}].q_weight", weights["q_weight"], (n_heads * head_dim, hidden))
        _expect_shape(f"layers[{index}].k_weight", weights["k_weight"], (n_kv * head_dim, hidden))
        _expect_shape(f"layers[{index}].v_weight", weights["v_weight"], (n_kv * head_dim, hidden))
        _expect_shape(f"layers[{index}].o_weight", weights["o_weight"], (hidden, n_heads * head_dim))
        _expect_shape(f"layers[{index}].gate_weight", weights["gate_weight"], (intermediate, hidden))
        _expect_shape(f"layers[{index}].up_weight", weights["up_weight"], (intermediate, hidden))
        _expect_shape(f"layers[{index}].down_weight", weights["down_weight"], (hidden, intermediate))
        layer_weights.append(weights)

    positions = torch.arange(seq, dtype=torch.float64)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    freqs = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)
    rope = torch.cat([freqs, freqs], dim=-1)
    cos, sin = rope.cos(), rope.sin()

    half = head_dim // 2

    def rotate(t):
        rotated = torch.cat([-t[..., half:], t[..., :half]], dim=-1)
        return t * cos + rotated * sin

    def split_heads(projected, count):
        return projected.reshape(batch, seq, count, head_dim).transpose(1, 2)

    q_index = torch.arange(seq).view(seq, 1)
    kv_index = torch.arange(seq).view(1, seq)
    causal = kv_index <= q_index
    causal_sliding = causal & (kv_index > q_index - window)

    repeat = n_heads // n_kv
    scale = pre_attn_scalar**-0.5

    x = embed[input_ids] * math.sqrt(hidden)
    for index in range(n_layers):
        weights = layer_weights[index]
        allowed = causal_sliding if layer_types[index] == "sliding_attention" else causal

        normed = _rms(x, weights["input_ln_weight"], eps)
        query = rotate(split_heads(normed @ weights["q_weight"].transpose(-1, -2), n_heads))
        key = rotate(split_heads(normed @ weights["k_weight"].transpose(-1, -2), n_kv))
        value = split_heads(normed @ weights["v_weight"].transpose(-1, -2), n_kv)
        if repeat > 1:
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)

        logits = (query @ key.transpose(-1, -2)) * scale
        if attn_softcap is not None:
            logits = attn_softcap * torch.tanh(logits / attn_softcap)
        logits = logits.masked_fill(~allowed, float("-inf"))
        attention = torch.softmax(logits, dim=-1)

        mixed = (attention @ value).transpose(1, 2).reshape(batch, seq, n_heads * head_dim)
        x = x + _rms(mixed @ weights["o_weight"].transpose(-1, -2), weights["post_attention_ln_weight"], eps)

        feedforward = _rms(x, weights["pre_feedforward_ln_weight"], eps)
        gate = torch.nn.functional.gelu(feedforward @ weights["gate_weight"].transpose(-1, -2), approximate="tanh")
        up = feedforward @ weights["up_weight"].transpose(-1, -2)
        x = x + _rms((gate * up) @ weights["down_weight"].transpose(-1, -2), weights["post_feedforward_ln_weight"], eps)

    x = _rms(x, final_norm, eps)
    final_logits = x @ head_weight.transpose(-1, -2)
    if final_softcap is not None:
        final_logits = final_softcap * torch.tanh(final_logits / final_softcap)
    return final_logits.to(torch.float32)
