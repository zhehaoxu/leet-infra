def solve(token_ids, audio_offset, end_token_id, block_size_list):
    text_tokens = []
    audio_tokens = []
    complete_tokens = []
    block_size_idx = 0
    block_size = block_size_list[0]
    is_finalize = False
    chunks = []
    for token_id in token_ids:
        if token_id == end_token_id:
            is_finalize = True
        if len(audio_tokens) >= block_size or (is_finalize and audio_tokens):
            chunks.append([list(audio_tokens), block_size, is_finalize])
            if block_size_idx < len(block_size_list) - 1:
                block_size_idx += 1
                block_size = block_size_list[block_size_idx]
            audio_tokens = []
        if not is_finalize:
            complete_tokens.append(token_id)
            if token_id >= audio_offset:
                audio_tokens.append(token_id - audio_offset)
            else:
                text_tokens.append(token_id)
    return {
        "complete_tokens": complete_tokens,
        "text_tokens": text_tokens,
        "chunks": chunks,
        "dropped_tail": audio_tokens,
        "final_block_size": block_size,
        "is_finalize": is_finalize,
    }
