"""KL-Loss (CVPR'19) variance voting + soft-NMS, official stdiou branch.

Reproduce `detectron/utils/py_cpu_nms.py::soft` (ethanhe42/KL-Loss,
cfg.STD.METHOD == 'stdiou'): greedy argmax pick, per-coordinate
variance-weighted box fusion of all remaining boxes with IoU > thresh,
soft score decay exp(-iou^2/0.5) on ANY positive overlap, tail-swap
removal below 0.001.

Ledger keys:
  ious    - initial N x N IoU matrix (legacy VOC +1 pixel convention)
  steps   - per-round: idx (original index of the picked box), box (fused
            coords), voters (original indices), weights (p_k / s_k^2)
  final   - kept boxes in processing order [x1, y1, x2, y2, score]
  removed - original indices decayed below the score floor
"""
from typing import Dict, List


def solve(dets: List[List[float]], confs: List[List[float]],
          thresh: float, iou_sigma: float) -> Dict:
    # TODO: implement the official stdiou soft-NMS + variance voting pass
    raise NotImplementedError
