import torch


def _oracle(ranks, k):
    L, N = ranks.shape
    sc = [0.0] * N
    for l in range(L):
        for d in range(N):
            r = int(ranks[l, d])
            if r > 0:
                sc[d] += 1.0 / (k + r)
    idx = list(range(N))
    idx.sort(key=lambda d: (-sc[d], d))
    return torch.tensor(idx, dtype=torch.long)


def check_example(solve):
    ranks = torch.tensor([[1, 2, 0], [2, 0, 1]])
    y = solve(ranks, 60)
    assert y.tolist() == [0, 2, 1], f"got {y.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    for ranks in [torch.tensor([[1, 2, 3, 0], [0, 1, 4, 2], [3, 0, 1, 2]]),
                  torch.tensor([[2, 0, 1], [1, 0, 3]]),
                  torch.tensor([[1], [2], [0]])]:
        y = solve(ranks, 60)
        assert torch.equal(y, _oracle(ranks, 60)), (ranks.tolist(), y.tolist())


def check_single_list_follows_rank(solve):
    # 单路：doc0 名次3, doc1 名次1, doc2 名次2 -> 融合序 = 名次升序 = [1,2,0]
    ranks = torch.tensor([[3, 1, 2]])
    y = solve(ranks, 60)
    assert y.tolist() == [1, 2, 0], y.tolist()


def check_absent_document_last(solve):
    # doc1 在两路都缺席(全0) -> score=0 -> 必排最后
    ranks = torch.tensor([[2, 0, 1], [1, 0, 3]])
    y = solve(ranks, 60)
    assert y[-1].item() == 1, y.tolist()


def check_permutation_and_dtype(solve):
    g = torch.Generator().manual_seed(31)
    ranks = torch.randint(0, 5, (4, 7), generator=g)
    y = solve(ranks, 60)
    assert y.dtype == torch.int64, y.dtype
    assert sorted(y.tolist()) == list(range(7)), y.tolist()
