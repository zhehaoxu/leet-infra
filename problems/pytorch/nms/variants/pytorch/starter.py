import torch


def solve(boxes, scores, iou_thresh):
    # TODO: NMS。按分数降序贪心：取队首框保留，抑制掉与它 IoU>iou_thresh 的候选，
    # 重复到候选清空。返回保留框原始索引（long，分数从高到低）。box=[x1,y1,x2,y2]。
    raise NotImplementedError
