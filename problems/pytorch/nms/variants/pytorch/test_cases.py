import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    boxes = torch.tensor([[0.0, 0, 10, 10], [1, 1, 11, 11],
                          [50, 50, 60, 60], [51, 51, 61, 61]])
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    keep = solve(boxes, scores, 0.5)
    assert keep.tolist() == [0, 2], keep.tolist()


def check_all_disjoint_keep_all(solve):
    # 互不相交的框全部保留，顺序按分数降序
    boxes = torch.tensor([[0.0, 0, 1, 1], [10, 10, 11, 11], [20, 20, 21, 21]])
    scores = torch.tensor([0.1, 0.9, 0.5])
    keep = solve(boxes, scores, 0.5)
    assert keep.tolist() == [1, 2, 0], keep.tolist()


def check_threshold_effect(solve):
    # 阈值越低抑制越强：thresh=0.01 时高重叠次框被抑制，只剩主框
    boxes = torch.tensor([[0.0, 0, 10, 10], [1, 1, 11, 11]])
    scores = torch.tensor([0.9, 0.8])
    assert solve(boxes, scores, 0.01).tolist() == [0]
    # 阈值很高（0.99）时 IoU≈0.82 < 0.99 不抑制，两者都留
    assert solve(boxes, scores, 0.99).tolist() == [0, 1]


def check_scores_desc_order(solve):
    torch.manual_seed(0)
    boxes = torch.rand(20, 4) * 100
    boxes = torch.stack([boxes[:, :2], boxes[:, 2:] + boxes[:, :2]], dim=2).reshape(20, 4)
    scores = torch.rand(20)
    keep = solve(boxes, scores, 0.5)
    kept_scores = scores[keep]
    assert bool((kept_scores[:-1] >= kept_scores[1:]).all()), "保留顺序应分数降序"
    assert len(keep.tolist()) == len(set(keep.tolist())), "索引应唯一"


check_basic.example = True
