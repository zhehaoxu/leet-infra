_DETECTION_STOP_TOKENS = {
    "</s>",
    "<|end_of_text|>",
    "<|eot_id|>",
    "<|endoftext|>",
    "<eos>",
    "<|eos|>",
    "<end_of_turn>",
    "<｜end▁of▁sentence｜>",
}


def _lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def solve(encoded_vocab, targets, uncertain_token_ids, stop_token_ids=None):
    if not isinstance(encoded_vocab, (list, tuple)) or any(
        not isinstance(t, str) for t in encoded_vocab
    ):
        raise ValueError("encoded_vocab must be a list of strings")
    if not isinstance(targets, (list, tuple)) or any(
        not isinstance(t, str) for t in targets
    ):
        raise ValueError("targets must be a list of strings")
    if stop_token_ids is not None:
        if isinstance(stop_token_ids, (str, bytes)) or not isinstance(
            stop_token_ids, (list, tuple, set, frozenset)
        ):
            raise ValueError("stop_token_ids must be a collection of ints")
        for i in stop_token_ids:
            if isinstance(i, bool) or not isinstance(i, int):
                raise ValueError("stop_token_ids must contain ints")
            if i < 0 or i >= len(encoded_vocab):
                raise ValueError("stop_token_ids contains an out-of-range id")
    if not isinstance(uncertain_token_ids, (list, tuple)):
        raise ValueError("uncertain_token_ids must be a list of ints")

    # === 第一层：TokenizerInfo 伪 trie ===
    if stop_token_ids is None:
        stop_ids = {
            i for i, tok in enumerate(encoded_vocab) if tok in _DETECTION_STOP_TOKENS
        }
    else:
        stop_ids = set(int(i) for i in stop_token_ids)

    pairs = []
    special_ids = set()
    for i, tok in enumerate(encoded_vocab):
        if i in stop_ids:
            continue
        if tok == "":
            special_ids.add(i)
            continue
        pairs.append((i, tok))
    # 官方 std::sort 按 token 字符串；并列（相同字符串）钉死按 id 升序
    pairs.sort(key=lambda p: (p[1], p[0]))
    sorted_token_ids = [p[0] for p in pairs]
    sorted_tokens = [p[1] for p in pairs]

    n_sorted = len(pairs)
    # 官方单调栈逐行转写；find 子串误判 quirk 原样保留
    trie_range = [0] * n_sorted
    stack = []
    for i in range(n_sorted):
        tok = sorted_tokens[i]
        while stack and tok.find(stack[-1][0]) == -1:
            _, j = stack.pop()
            trie_range[j] = i
        stack.append((tok, i))
    while stack:
        _, j = stack.pop()
        trie_range[j] = n_sorted

    id2sorted = {}
    for idx, p in enumerate(pairs):
        id2sorted[p[0]] = idx

    # === 第二层：前缀增量复用账 ===
    u_idx = []
    for tid in uncertain_token_ids:
        if isinstance(tid, bool) or not isinstance(tid, int):
            raise ValueError("uncertain_token_ids must contain ints")
        if tid < 0 or tid >= len(encoded_vocab):
            raise ValueError("uncertain_token_ids contains an out-of-range id")
        if tid in stop_ids or tid in special_ids:
            raise ValueError("uncertain_token_ids must not contain stop/special tokens")
        u_idx.append(id2sorted[tid])
    u_idx.sort()

    target_list = list(targets)

    def advance(s):
        return any(t.startswith(s) for t in target_list)

    accepted_token_ids = []
    ledger = []
    prev_token = None
    prev_matched = 0
    last_rejected_range = 0
    for idx in u_idx:
        cur = sorted_tokens[idx]
        tid = sorted_token_ids[idx]
        if idx < last_rejected_range:
            # 官方 kRejected delta：整段子树直接拒
            ledger.append({
                "token_id": tid, "sorted_index": idx,
                "branch": "subtree-skip", "lcp": -1, "matched_from": -1,
            })
            continue
        accepted = True
        branch = None
        lcp = -1
        if prev_token is not None:
            lcp = _lcp(cur, prev_token)
            if lcp > prev_matched:
                # 状态机确定性：同前缀必同失败 → 短路拒整棵子树
                last_rejected_range = trie_range[idx]
                accepted = False
                branch = "lcp-short-circuit"
            elif lcp < prev_matched:
                prev_matched = lcp  # 官方 PopLastStates(prev_matched - lcp)
        matched_from = -1
        if accepted:
            matched_from = prev_matched
            for j in range(matched_from, len(cur)):
                if not advance(cur[:j + 1]):
                    last_rejected_range = trie_range[idx]
                    accepted = False
                    branch = "walk-rejected"
                    break
                prev_matched = j + 1
            if accepted:
                branch = "accepted"
                accepted_token_ids.append(tid)
        ledger.append({
            "token_id": tid, "sorted_index": idx, "branch": branch,
            "lcp": lcp, "matched_from": matched_from,
        })
        # 官方：prev_token 每轮更新（接受/拒绝都更新）
        prev_token = cur

    return {
        "sorted_token_ids": sorted_token_ids,
        "sorted_tokens": sorted_tokens,
        "trie_subtree_nodes_range": trie_range,
        "stop_token_ids": sorted(stop_ids),
        "special_token_ids": sorted(special_ids),
        "accepted_token_ids": accepted_token_ids,
        "ledger": ledger,
    }
