def solve(prompt_len, max_num_batched_tokens, n_decode_running):
    """Chunked prefill：avail = budget - n_decode_running；avail<=0 返回 []，
    否则每步切 min(剩余, avail)，返回各步 chunk 大小 list[int]。"""
    # TODO
    raise NotImplementedError
