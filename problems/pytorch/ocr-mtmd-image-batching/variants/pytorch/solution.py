def solve(chunks, n_batch, logits_last=False, n_past=0):
    if not isinstance(chunks, (list, tuple)):
        raise ValueError("chunks must be a list of chunk dicts")
    if isinstance(n_batch, bool) or not isinstance(n_batch, int) or n_batch <= 0:
        raise ValueError("n_batch must be a positive int")
    if not isinstance(logits_last, bool):
        raise ValueError("logits_last must be a bool")
    if isinstance(n_past, bool) or not isinstance(n_past, int) or n_past < 0:
        raise ValueError("n_past must be a non-negative int")
    for ch in chunks:
        if not isinstance(ch, dict):
            raise ValueError("each chunk must be a dict")
        if ch.get("type") not in ("text", "image"):
            raise ValueError("chunk type must be 'text' or 'image'")
        nt = ch.get("n_tokens")
        if isinstance(nt, bool) or not isinstance(nt, int) or nt < 0:
            raise ValueError("chunk n_tokens must be a non-negative int")

    batches = []
    logits_pos = None
    cur = n_past
    n_chunks = len(chunks)
    for ci, ch in enumerate(chunks):
        n_tokens = ch["n_tokens"]
        if ch["type"] == "text":
            i = 0
            while i < n_tokens:
                n_added = 0
                pos_start = cur
                while i < n_tokens and n_added < n_batch:
                    i += 1
                    cur += 1
                    n_added += 1
                is_last_token = i == n_tokens
                chunk_is_last = ci == n_chunks - 1
                take_logits = bool(logits_last and chunk_is_last and is_last_token)
                if take_logits:
                    logits_pos = pos_start + n_added - 1
                batches.append(
                    {
                        "chunk_index": ci,
                        "kind": "text",
                        "n_tokens": n_added,
                        "pos_start": pos_start,
                        "logits": take_logits,
                    }
                )
        else:  # image
            n_img_batches = (n_tokens + n_batch - 1) // n_batch
            for b in range(n_img_batches):
                pos_offset = b * n_batch
                n_tokens_batch = min(n_batch, n_tokens - pos_offset)
                batches.append(
                    {
                        "chunk_index": ci,
                        "kind": "image",
                        "n_tokens": n_tokens_batch,
                        "pos_start": cur + pos_offset,
                        "logits": False,
                    }
                )
            # 非 M-RoPE：n_pos == n_tokens，chunk 解码完后一次推进
            cur += n_tokens
    return {"batches": batches, "new_n_past": cur, "logits_pos": logits_pos}
