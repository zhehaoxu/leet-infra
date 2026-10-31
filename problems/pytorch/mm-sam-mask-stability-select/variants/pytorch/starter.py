import torch


def solve(masks, iou_preds, num_points, mask_threshold=0.0, offset=1.0):
    """中文契约 / English contract
    masks: float32/float64 (B,C,H,W) 原始 decoder 掩码 logits，可非连续，不可修改 /
        raw decoder mask logits; may be noncontiguous, must not be mutated.
    iou_preds: 同 dtype (B,C) 原始 IoU 预测，只用于回传账目 /
        same dtype (B, C) raw IoU predictions, returned only as a ledger.
    num_points: 正 Python int（点提示数量，上游传 point_coords.shape[1]）；
        bool/float/<=0 → ValueError / positive Python int (bool and float excluded).
    mask_threshold: Python 实数，默认 0.0；offset: Python 实数，默认 1.0，必须 >0；
        bool、非实数、非有限或 offset<=0 → ValueError / finite Python reals, offset > 0.
    B=1..4, C=1..5, H,W=1..8, num_points=1..12；全部值有限，绝对值 <=1e4 / finite values.
    masks/iou_preds 同 dtype、同 device，形状不匹配、非 4 维、含 NaN/Inf → ValueError；
        其余输入保证合法 / shape or dtype mismatch, non-4-D masks, NaN/Inf → ValueError.
    稳定分数 / stability: inter/union，inter = (masks > thr+offset) 的逐行求和再求总，
        union = (masks > thr-offset) 同理，两个计数都取整数；union==0 时约定 0.0 /
        integer counts; stability is defined as 0.0 when the union is empty.
    偏置 / bias: biased[b,c] = stability[b,c] + (num_points - 2.5) * (1000 if c == 0 else 0)；
        best_idx[b] = biased[b] 的最大下标，同值取最小下标（torch argmax，不是稳定排序）/
        first index wins on ties.
    返回 / Returns (selected_mask bool (B,H,W), selected_score (B,), stability (B,C),
        biased (B,C), best_idx int64 (B,), areas int64 (B,C), pred_iou_selected (B,))；
        selected_mask[b] = masks[b, best_idx[b]] > mask_threshold；
        selected_score[b] = stability[b, best_idx[b]]；
        areas[b,c] = (masks[b,c] > mask_threshold) 的像素数（与选择用同一阈值）；
        pred_iou_selected[b] = iou_preds[b, best_idx[b]]。
    浮点输出保持 masks dtype，整型输出为 int64，全部同 device /
        float outputs reuse the masks dtype; integer outputs are int64.
    禁止 import segment_anything/numpy 掩码工具；只用基础 torch 算子 / no SAM libraries.
    """
    raise NotImplementedError
