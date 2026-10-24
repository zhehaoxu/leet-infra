def solve(prompt_len, max_num_batched_tokens, n_decode_running):
    """Chunked prefill 分块调度。"""
    prompt_len = int(prompt_len)
    avail = int(max_num_batched_tokens) - int(n_decode_running)
    chunks = []
    if avail <= 0:
        return chunks
    rem = prompt_len
    while rem > 0:
        step_chunk = min(rem, avail)
        chunks.append(step_chunk)
        rem -= step_chunk
    return chunks
