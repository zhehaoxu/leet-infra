import math
import torch


def _oracle(input_ids, beam_scores, next_log_probs, num_beams):
    """独立纯 python：枚举 B*V 候选、按 (-score, flat_index) 排序取前 k，手算回退。"""
    B = int(next_log_probs.shape[0])
    V = int(next_log_probs.shape[1])
    ii = input_ids.tolist()
    bs = beam_scores.tolist()
    nlp = next_log_probs.tolist()
    cands = []
    for b in range(B):
        for t in range(V):
            cands.append((bs[b] + nlp[b][t], b * V + t))
    cands.sort(key=lambda x: (-x[0], x[1]))
    chosen = cands[:int(num_beams)]
    new_ids = []
    new_scores = []
    for sc, fi in chosen:
        b = fi // V
        t = fi % V
        new_ids.append(ii[b] + [t])
        new_scores.append(sc)
    return torch.tensor(new_ids, dtype=torch.long), torch.tensor(new_scores)


def check_example(solve):
    input_ids = torch.tensor([[1, 2], [1, 3], [4, 5]])
    beam_scores = torch.tensor([-0.1, -0.2, -1.5])
    nlp = torch.log(torch.tensor([
        [0.1, 0.4, 0.3, 0.2],
        [0.5, 0.1, 0.2, 0.2],
        [0.2, 0.3, 0.4, 0.1],
    ]))
    nid, nsc = solve(input_ids, beam_scores, nlp, 2)
    assert nid.tolist() == [[1, 3, 0], [1, 2, 1]], nid.tolist()
    assert [round(x, 6) for x in nsc.tolist()] == [-0.893147, -1.016291], nsc.tolist()


def check_greedy_k1(solve):
    # k=1 退化为全局最优单条
    input_ids = torch.tensor([[7], [8]])
    beam_scores = torch.tensor([-0.5, -0.1])
    nlp = torch.log(torch.tensor([[0.1, 0.9], [0.9, 0.1]]))
    nid, nsc = solve(input_ids, beam_scores, nlp, 1)
    assert nid.tolist() == [[8, 0]], nid.tolist()


def check_matches_oracle(solve):
    torch.manual_seed(5)
    for _ in range(30):
        B = int(torch.randint(2, 6, (1,)))
        L = int(torch.randint(1, 4, (1,)))
        V = int(torch.randint(2, 8, (1,)))
        k = int(torch.randint(1, B + 1, (1,)))
        input_ids = torch.randint(0, 100, (B, L))
        beam_scores = -torch.rand(B) * 3
        nlp = torch.log(torch.rand(B, V) + 1e-3)
        nid, nsc = solve(input_ids, beam_scores, nlp, k)
        oid, osc = _oracle(input_ids, beam_scores, nlp, k)
        assert nid.tolist() == oid.tolist(), (k, nid.tolist(), oid.tolist())
        for a, b in zip(nsc.tolist(), osc.tolist()):
            assert math.isclose(a, b, rel_tol=1e-5), (a, b)


check_example.example = True
