def _check_int(name, value, lo=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是 int")
    if lo is not None and value < lo:
        raise ValueError(f"{name} 必须 >= {lo}")
    return value


def solve(ids, initial_vocab_size, max_codebook_size, max_subtokens,
          pad_token_id, disabled_ids=None, truncation=False, max_length=None,
          padding="do_not_pad"):
    """zip2zip LZW 动态码本编码（2610.13605，CPU 参考口径）。"""
    if not isinstance(ids, (list, tuple)) or not ids:
        raise ValueError("ids 必须是非空 list")
    initial_vocab_size = _check_int("initial_vocab_size", initial_vocab_size, 1)
    max_codebook_size = _check_int("max_codebook_size", max_codebook_size, 1)
    max_subtokens = _check_int("max_subtokens", max_subtokens, 2)
    pad_token_id = _check_int("pad_token_id", pad_token_id, 0)
    if pad_token_id < initial_vocab_size + max_codebook_size:
        raise ValueError("pad_token_id 必须不与任何可发射 id 重叠")
    seq = []
    for i, x in enumerate(ids):
        x = _check_int(f"ids[{i}]", x, 0)
        if x >= initial_vocab_size:
            raise ValueError(f"ids[{i}] 必须小于 initial_vocab_size")
        seq.append(x)
    disabled = []
    if disabled_ids is not None:
        if not isinstance(disabled_ids, (list, tuple)):
            raise ValueError("disabled_ids 必须是 list 或 None")
        seen = set()
        for i, x in enumerate(disabled_ids):
            x = _check_int(f"disabled_ids[{i}]", x, 0)
            if x >= initial_vocab_size:
                raise ValueError("disabled_ids 必须落在基词表内")
            if x in seen:
                raise ValueError("disabled_ids 不能重复")
            seen.add(x)
            disabled.append(x)
    disabled_set = set(disabled)
    if isinstance(truncation, bool) is False:
        raise ValueError("truncation 必须是 bool")
    if max_length is not None:
        max_length = _check_int("max_length", max_length, 1)
    if truncation and max_length is None:
        raise ValueError("truncation=True 时必须给 max_length")
    if padding not in ("do_not_pad", "max_length"):
        raise ValueError("padding 只能是 'do_not_pad' 或 'max_length'")
    if padding == "max_length" and max_length is None:
        raise ValueError("padding='max_length' 时必须给 max_length")

    codebook = {}          # phrase(tuple) -> hyper id
    buffer = []
    next_id = initial_vocab_size
    limit = initial_vocab_size + max_codebook_size
    compressed = []

    def emit():
        if truncation and len(compressed) >= max_length:
            return
        if len(buffer) == 1:
            compressed.append(buffer[0])
        else:
            compressed.append(codebook[tuple(buffer)])

    i = 0
    while i < len(seq):
        if truncation and len(compressed) >= max_length:
            break
        token = seq[i]
        i += 1

        if token in disabled_set:
            if buffer:
                emit()
                buffer.clear()
            buffer.append(token)
            emit()
            buffer.clear()
            continue

        buffer.append(token)
        known = len(buffer) == 1 or tuple(buffer) in codebook
        if not known:
            if next_id < limit:
                codebook[tuple(buffer)] = next_id
                next_id += 1
            buffer.pop()
            emit()
            buffer.clear()
            buffer.append(token)
        if len(buffer) == max_subtokens:
            emit()
            buffer.clear()

    if buffer:
        emit()

    if padding == "max_length":
        if len(compressed) > max_length:
            raise ValueError("压缩结果超过 max_length（未开 truncation）")
        if len(compressed) < max_length:
            compressed = [pad_token_id] * (max_length - len(compressed)) + compressed

    attention_mask = [0 if c == pad_token_id else 1 for c in compressed]
    codebook_out = {
        hyper: list(phrase)
        for phrase, hyper in sorted(codebook.items(), key=lambda kv: kv[1])
    }
    return {
        "compressed_ids": compressed,
        "attention_mask": attention_mask,
        "codebook": codebook_out,
    }
