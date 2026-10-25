import math

import torch


def _oracle(img, txt, scale):
    n = img.shape[0]
    li = (scale * (img @ txt.T)).double()
    lt = (scale * (txt @ img.T)).double()
    tot = 0.0
    for row in (li, lt):
        for i in range(n):
            m = row[i].max().item()
            lse = m + math.log(sum(math.exp(row[i, j].item() - m) for j in range(n)))
            tot += -(row[i, i].item() - lse)
    return 0.5 * (tot / n)


def _norm(x):
    return x / x.norm(dim=1, keepdim=True)


def check_example(solve):
    im = torch.tensor([[1.0, 0.0], [0.6, 0.8]])
    tx = torch.tensor([[1.0, 0.0], [0.2, 0.98]])
    y = solve(im, tx, 4.0)
    assert abs(y.item() - 0.13539454340934753) < 1e-5, f"got {y.item()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(31)
    for n, d, s in [(4, 8, 4.0), (6, 5, 1.0), (3, 3, 10.0)]:
        im = _norm(torch.randn(n, d))
        tx = _norm(torch.randn(n, d))
        assert abs(solve(im, tx, s).item() - _oracle(im, tx, s)) < 1e-5


def check_perfect_alignment_low_loss(solve):
    # 图像=文本、彼此正交、温度很大 → softmax 命中对角 → 损失趋 0
    feats = torch.eye(4)
    y = solve(feats, feats, 50.0)
    assert y.item() < 1e-4, y.item()


def check_symmetry(solve):
    torch.manual_seed(32)
    im = _norm(torch.randn(5, 6))
    tx = _norm(torch.randn(5, 6))
    assert abs(solve(im, tx, 3.0).item() - solve(tx, im, 3.0).item()) < 1e-5


def check_returns_scalar(solve):
    im = _norm(torch.randn(3, 4))
    tx = _norm(torch.randn(3, 4))
    y = solve(im, tx, 2.0)
    assert y.dim() == 0 and y.dtype == torch.float32, (y.dim(), y.dtype)
