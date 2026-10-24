import math
import torch


def _ls(x):
    return min(0.0, x) - math.log1p(math.exp(-abs(x)))


def _oracle(ch, rh, cm, rm, w, b):
    """独立纯 python：逐行找最大 mask=1 下标、手算点积与 logsigmoid。"""
    def pool(hidden, mask):
        res = []
        for bi in range(len(hidden)):
            idx = -1
            for j in range(len(mask[bi])):
                if mask[bi][j] == 1:
                    idx = j
            row = hidden[bi][idx]
            res.append(sum(row[d] * w[d] for d in range(len(w))) + b)
        return res
    rc = pool(ch, cm)
    rr = pool(rh, rm)
    return sum(-_ls(rc[i] - rr[i]) for i in range(len(rc))) / len(rc)


def check_example(solve):
    ch = torch.tensor([[[0.1, 0.3], [0.2, -0.1], [0.5, 0.2]],
                        [[0.3, 0.0], [0.9, 0.4], [0.0, 0.0]]])
    rh = torch.tensor([[[0.4, 0.1], [0.2, 0.5], [0.1, -0.2]],
                        [[0.6, 0.2], [0.7, -0.3], [0.0, 0.0]]])
    cm = torch.tensor([[1, 1, 1], [1, 1, 0]])
    rm = torch.tensor([[1, 1, 0], [1, 1, 1]])
    w = torch.tensor([1.0, 0.5])
    out = solve(ch, rh, cm, rm, w, 0.1)
    assert abs(out.item() - 0.454146) < 1e-5, out.item()


def check_noncontiguous_mask(solve):
    # mask 中间有洞，池化应取最大有效下标（非最后一个 1 之后的位置）
    ch = torch.zeros(1, 4, 1)
    ch[0, 0, 0] = 1.0
    ch[0, 3, 0] = 5.0  # 最后有效
    rh = torch.zeros(1, 4, 1)
    rh[0, 2, 0] = 2.0  # mask idx2 最后有效
    cm = torch.tensor([[1, 0, 0, 1]])
    rm = torch.tensor([[0, 0, 1, 0]])
    out = solve(ch, rh, cm, rm, torch.tensor([1.0]), 0.0)
    # r_c=5, r_r=2, diff=3 -> -logsigmoid(3)= -log(0.95257)=0.048587
    assert abs(out.item() - (-_ls(3.0))) < 1e-5, out.item()


def check_matches_oracle(solve):
    torch.manual_seed(4)
    for _ in range(40):
        N = int(torch.randint(1, 5, (1,)))
        L = int(torch.randint(1, 6, (1,)))
        D = int(torch.randint(1, 5, (1,)))
        ch = torch.randn(N, L, D)
        rh = torch.randn(N, L, D)
        cm = (torch.rand(N, L) > 0.3).float()
        rm = (torch.rand(N, L) > 0.3).float()
        cm[:, 0] = 1  # 保证每行至少一个有效
        rm[:, 0] = 1
        w = torch.randn(D)
        b = float(torch.randn(1))
        got = float(solve(ch, rh, cm, rm, w, b).item())
        exp = _oracle(ch.tolist(), rh.tolist(), cm.tolist(), rm.tolist(), w.tolist(), b)
        assert abs(got - exp) < 1e-4, (got, exp)


check_example.example = True
