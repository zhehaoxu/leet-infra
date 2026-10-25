import torch

VOCAB = {
    "<unk>": 0,
    "the": 1, "cat": 2, "sat": 3, "on": 4, "mat": 5, "a": 6, "dog": 7, "is": 8,
    "nice": 9, "Question:": 10, "Answer:": 11, "True": 12, "False": 13,
    "yes": 14, "no": 15, "here": 16, "there": 17,
}

DEMOS = [
    "the cat sat on the mat",     # 6 个 token（全在词表里）
    "the dog is very nice",       # "very" 不在词表 -> 4 个字符 token，共 8
    "yes",                        # 1
    "no",                         # 1
]
QUERY = "Question:"
TAIL = "Answer:"
BUDGET = 15
EXAMPLE_KEPT = [0]       # 参考解跑出：预算 15 只够第 0 条演示
EXAMPLE_LEN = 8          # 参考解跑出的最终 prompt token 数


def _toy_tokenize(text):
    """独立重算：手写扫描器切空白 + 逐字符展开 OOV 词，不用 str.split。"""
    words = []
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        j = i
        while j < n and not text[j].isspace():
            j += 1
        if j > i:
            words.append(text[i:j])
        i = j
    count = 0
    for w in words:
        if w in VOCAB:
            count += 1
        else:
            count += len(w)
    return count


def _oracle(demos, query, tail, budget):
    used = _toy_tokenize(query) + _toy_tokenize(tail)
    kept = []
    i = 0
    while i < len(demos):                 # while 循环 + 显式 break 标记，与参考解结构不同
        cost = _toy_tokenize(demos[i])
        if used + cost > budget:
            break
        kept.append(i)
        used += cost
        i += 1
    return kept, used


def check_example(solve):
    kept, final_len = solve(DEMOS, QUERY, TAIL, BUDGET)
    assert torch.equal(kept, torch.tensor(EXAMPLE_KEPT, dtype=torch.long)), kept.tolist()
    assert final_len == EXAMPLE_LEN, final_len


check_example.example = True


def check_matches_oracle(solve):
    cases = [
        (DEMOS, QUERY, TAIL, 15),
        (DEMOS, QUERY, TAIL, 100),
        (DEMOS, QUERY, TAIL, 2),
        (["here", "the cat"], "Question:", "Answer:", 6),
        ([], "Question:", "Answer:", 0),
        (["the mat is on the mat", "yes"], "a dog is nice", "no", 12),
    ]
    for demos, query, tail, budget in cases:
        kept, length = solve(demos, query, tail, budget)
        ekept, elen = _oracle(demos, query, tail, budget)
        assert torch.equal(kept, torch.tensor(ekept, dtype=torch.long)), (demos, budget, kept.tolist(), ekept)
        assert length == elen, (demos, budget, length, elen)


def check_random_against_oracle(solve):
    g = torch.Generator().manual_seed(37)
    pool = ["the", "cat", "sat", "on", "mat", "dog", "is", "nice", "yes", "no",
            "here", "there", "Question:", "Answer:", "unknownword", "x"]
    for _ in range(25):
        m = int(torch.randint(0, 6, (1,), generator=g))
        demos = []
        for _ in range(m):
            words = int(torch.randint(1, 6, (1,), generator=g))
            demos.append(" ".join(pool[int(x)] for x in torch.randint(0, len(pool), (words,), generator=g)))
        query = " ".join(pool[int(x)] for x in torch.randint(0, len(pool), (2,), generator=g))
        tail = pool[int(torch.randint(0, len(pool), (1,), generator=g))]
        budget = int(torch.randint(0, 30, (1,), generator=g))
        kept, length = solve(demos, query, tail, budget)
        ekept, elen = _oracle(demos, query, tail, budget)
        assert torch.equal(kept, torch.tensor(ekept, dtype=torch.long)), (demos, query, tail, budget)
        assert length == elen, (demos, query, tail, budget, length, elen)


def check_assembled_length_is_real(solve):
    # final_len 必须等于「真的把选中的演示拼出来再分词」得到的 token 数
    for budget in (0, 2, 5, 8, 13, 15, 4096):
        kept, length = solve(DEMOS, QUERY, TAIL, budget)
        assembled = " ".join([DEMOS[i] for i in kept.tolist()] + [QUERY, TAIL])
        assert length == _toy_tokenize(assembled), (budget, kept.tolist(), length, assembled)
        assert length >= 2, (budget, length)   # query 与 tail 永在


def check_exact_budget_boundary_keeps_demo(solve):
    # 恰好用满预算（== budget）时该演示必须被保留
    demos = ["the cat sat", "yes"]
    kept, length = solve(demos, "Question:", "Answer:", 6)
    assert torch.equal(kept, torch.tensor([0, 1], dtype=torch.long)), kept.tolist()
    assert length == 6, length
    kept2, length2 = solve(demos, "Question:", "Answer:", 5)
    assert torch.equal(kept2, torch.tensor([0], dtype=torch.long)), kept2.tolist()
    assert length2 == 5, length2
    kept3, length3 = solve(demos, "Question:", "Answer:", 4)
    assert kept3.numel() == 0 and length3 == 2, (kept3.tolist(), length3)


def check_hard_cut_does_not_skip_ahead(solve):
    # 第 1 条超预算 -> 直接截断；第 2、3 条虽短也不许回头捡
    demos = ["the cat", "unknownword unknownword", "yes", "no"]
    kept, length = solve(demos, "Question:", "Answer:", 5)
    assert torch.equal(kept, torch.tensor([0], dtype=torch.long)), kept.tolist()
    assert length == 4, length


def check_tail_always_survives_even_if_over_budget(solve):
    # query + tail 本身就超预算：仍然一条演示都不放，tail 不被裁掉，长度如实报超支
    demos = ["the", "cat"]
    kept, length = solve(demos, "Question:", "Answer:", 1)
    assert kept.numel() == 0, kept.tolist()
    assert length == 2, length


def check_no_demos(solve):
    # 空演示集（候选集为空）
    kept, length = solve([], QUERY, TAIL, 4096)
    assert kept.numel() == 0 and kept.dtype == torch.long, kept
    assert length == 2, length


def check_all_demos_kept_when_budget_is_ample(solve):
    kept, length = solve(DEMOS, QUERY, TAIL, 4096)
    assert torch.equal(kept, torch.tensor([0, 1, 2, 3], dtype=torch.long)), kept.tolist()
    assert length == 2 + 6 + 8 + 1 + 1, length       # query+tail 再叠四条演示


def check_oov_words_explode_cost(solve):
    # OOV 词按字符展开：'here' 在词表里只值 1，'HERE' 不在词表里值 4
    assert _toy_tokenize("here") == 1
    assert _toy_tokenize("HERE") == 4
    kept, length = solve(["here there", "HERE"], "Question:", "Answer:", 5)
    assert torch.equal(kept, torch.tensor([0], dtype=torch.long)), kept.tolist()
    assert length == 4, length
