import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    src = torch.tensor([[10, 20, 30], [40, 50, 60]])
    index = torch.tensor([[2, 0, 1]])
    y = solve(src, index, dim=1)
    expected = [[30, 10, 20]]
    assert torch.equal(y, torch.tensor(expected)), f"got {y.tolist()}"


def check_dim0(solve):
    torch.manual_seed(0)
    src = torch.randn(3, 4)
    index = torch.tensor([[0, 1, 2, 0], [2, 2, 1, 1]])
    assert torch.equal(solve(src, index, 0), torch.gather(src, 0, index))


def check_dim1_matches_torch(solve):
    torch.manual_seed(1)
    src = torch.randn(2, 5)
    index = torch.randint(0, 5, (2, 3))
    assert torch.equal(solve(src, index, 1), torch.gather(src, 1, index))


def check_3d(solve):
    torch.manual_seed(2)
    src = torch.randn(2, 3, 4)
    for dim in range(3):
        index = torch.randint(0, src.shape[dim], (2, 3, 4))
        assert torch.equal(solve(src, index, dim), torch.gather(src, dim, index)), f"dim={dim} 不一致"


def check_negative_dim(solve):
    torch.manual_seed(3)
    src = torch.randn(4, 6)
    index = torch.randint(0, 6, (4, 2))
    assert torch.equal(solve(src, index, -1), torch.gather(src, -1, index)), "负 dim 应等价最后一维"


def check_topk_semantics(solve):
    # 典型用法：按每行 top-k 的列索引取回最大值
    src = torch.tensor([[3.0, 1.0, 2.0], [0.5, 9.0, 4.0]])
    idx = src.argmax(dim=1, keepdim=True)
    y = solve(src, idx, dim=1)
    assert torch.equal(y, torch.tensor([[3.0], [9.0]])), "应取到每行最大值"


check_basic.example = True
